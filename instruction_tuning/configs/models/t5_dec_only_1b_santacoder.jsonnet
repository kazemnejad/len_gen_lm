{
    model+: {
        type: 'custom_decoder_only_t5_no_resize',
        hf_model_name: 'experiments/t5_dec_only_1b_santacoder_' + $.model.position_encoding_type,
        config+: {
            type: 'seq2seq_t5',
            hf_model_name: 'experiments/t5_dec_only_1b_santacoder_' + $.model.position_encoding_type,
        },
        from_pretrained: true,
        full_pretrained_path: 't5_dec_only_1b_santacoder_' + $.model.position_encoding_type,
    },
}
