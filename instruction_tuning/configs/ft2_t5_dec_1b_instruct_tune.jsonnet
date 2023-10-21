(import 'base.jsonnet')
+ (import 'models/t5_dec_only_1b_santacoder.jsonnet')
+ (import 'tokenizers/pretrained_fast.jsonnet')
+ (import 'trainer/base_model.jsonnet')
+ (import 'data/seq2seq.jsonnet')
+ (import 'data/seq2seq_instruct.jsonnet')
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
        decoder_only_input_output_sep_token: '\n',

        validation_portion: 1.0,
    },
    trainer: {
        type: 'decoder_only',

        bf16: true,
        bf16_full_eval: true,

        dataloader_num_workers: 8,
        dataloader_pin_memory: true,

        evaluation_strategy: 'steps',
        save_strategy: 'steps',
        logging_strategy: 'steps',

        save_steps: 300,
        eval_steps: 300,
        logging_steps: 20,
        save_total_limit: 5,


        learning_rate: 5e-05,
        lr_scheduler_type: 'cosine',
        warmup_ratio: 0.02,
        weight_decay: 0.001,
        num_train_epochs: 30,

        metric_for_best_model: 'seq_acc',

        predict_with_generate: true,
        generation_max_length: 1700,
        generation_num_beams: 1,

        auto_compute_batch_size: true,

        per_device_eval_batch_size: 16,
        per_device_train_batch_size: 2,
        gradient_accumulation_steps: 4,
    },
    analyzers: [
        (import 'analyzers/instruct_analyzer.jsonnet'),
    ],
}
