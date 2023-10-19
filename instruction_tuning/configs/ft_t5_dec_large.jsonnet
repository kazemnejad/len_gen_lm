(import 'base.jsonnet')
+ (import 'models/t5_dec_only_300m_wikitext.jsonnet')
+ (import 'tokenizers/pretrained_fast.jsonnet')
+ (import 'trainer/base_model.jsonnet')
+ (import 'data/seq2seq.jsonnet')
+ (import 'data/seq2seq_bos.jsonnet')
+ {
    global_vars+: {
        debug_mode: false,
    },
    dataset+: {
        is_decoder_only: true,
        decoder_only_block_size: 128,
        decoder_only_group_samples: false,
        decoder_only_mask_inputs: true,
        decoder_only_padding_side: 'right',
        decoder_only_include_position_ids: false,

        validation_portion: 0.5,
    },
    trainer+: {
        type: 'decoder_only',
    },
    analyzers: [
        (import 'analyzers/s2s_analyzer.jsonnet'),
    ],
}
