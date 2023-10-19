{
    dataset+: {
        name: 'octa',
        split: std.extVar('APP_DS_SPLIT'),
        train_filename: 'train.jsonl',
        validation_filename: 'validation.jsonl',
        test_filename: 'test.jsonl',

        source_seq_key: 'query',
        target_seq_key: 'response',

        decoder_only_input_output_sep_token: "",

        max_source_length: 10000,
        max_target_length: 10000,

        instance_processor+: {
            type: 'octa',
        },
    },
}
