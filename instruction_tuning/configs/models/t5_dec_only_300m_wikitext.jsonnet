{
    model+: {
        type: 'custom_decoder_only_t5_no_resize',
        hf_model_name: 'experiments/t5_dec_only_300m_wikitext_' + $.model.position_encoding_type + '/checkpoints',
        config+: {
            type: 'seq2seq_t5',
            hf_model_name: 'experiments/t5_dec_only_300m_wikitext_' + $.model.position_encoding_type + '/checkpoints',
        },
        from_pretrained: true,
        pretrained_path: 't5_dec_only_300m_wikitext_' + $.model.position_encoding_type,
    },
}
