#include "models.h"

ggml_cgraph * clip_graph_ling3vl::build() {
    // same as qwen3vl, but Ling has no patch bias
    GGML_ASSERT(model.patch_bias == nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);
    GGML_ASSERT(model.mm_input_norm_w != nullptr); // merger norm (pre spatial merge)

    const int batch_size = 1;
    const int n_pos      = n_patches;

    norm_type norm_t = NORM_TYPE_NORMAL;

    // vision M-RoPE with 2 sections (row, col); the temporal section is unused
    int mrope_sections[4] = {0, d_head/2, d_head/2, 0};

    ggml_tensor * inp = build_inp_with_temporal_merge();

    // spatial merge
    {
        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
        inp = ggml_cont_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(
            ctx0, inp,
            n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // per-patch merger norm before the spatial merge (merger.norm, LayerNorm)
    inp = build_norm(inp, model.mm_input_norm_w, model.mm_input_norm_b, norm_t, eps, -1);
    cb(inp, "merger_norm", -1);

    // calculate absolute position embedding and apply
    ggml_tensor * learned_pos_embd = resize_position_embeddings(GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
    learned_pos_embd = ggml_cont_4d(
        ctx0, learned_pos_embd,
        n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
    learned_pos_embd = ggml_reshape_4d(
        ctx0, learned_pos_embd,
        n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
    learned_pos_embd = ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3);
    learned_pos_embd = ggml_cont_3d(
        ctx0, learned_pos_embd,
        n_embd, n_patches_x * n_patches_y, batch_size);
    inp = ggml_add(ctx0, inp, learned_pos_embd);
    cb(inp, "inp_pos_emb", -1);

    ggml_tensor * inpL = inp;

    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
    }

    for (int il = 0; il < n_layer; il++) {
        auto & layer = model.layers[il];

        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "ln1", il);

        // self-attention
        {
            cur = build_mm(layer.qkv_w, cur);
            cur = ggml_add(ctx0, cur, layer.qkv_b);

            ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ 0);

            ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ ggml_row_size(cur->type, n_embd));

            ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ ggml_row_size(cur->type, 2 * n_embd));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // apply M-RoPE
            Qcur = ggml_rope_multi(
                ctx0, Qcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur = ggml_rope_multi(
                ctx0, Kcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        cb(cur, "ffn_inp", il);

        // layernorm2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // multimodal projection (linear_proj MLP over the merged patches)
    ggml_tensor * embeddings = inpL;
    embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);

    embeddings = build_ffn(embeddings,
        model.mm_0_w, model.mm_0_b,
        nullptr, nullptr,
        model.mm_1_w, model.mm_1_b,
        ffn_op_type::FFN_GELU, -1);

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}
