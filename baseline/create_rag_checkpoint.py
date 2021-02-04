import argparse

parser = argparse.ArgumentParser()
parser.add_argument("generator_model_name_or_path", type=str)
parser.add_argument("encoder_model_name_or_path", type=str)
parser.add_argument("checkpoint_path", type=str)
args, additional_args = parser.parse_known_args()

from transformers.configuration_auto import AutoConfig
from baseline.transformers.modeling_rag import PreTrainedRagModel
from baseline.transformers.configuration_rag import RagConfig

generator_config = AutoConfig.from_pretrained(args.generator_model_name_or_path)
config = RagConfig(
    pretrained_question_encoder_tokenizer_name_or_path=args.encoder_model_name_or_path if len(args.encoder_model_name_or_path) > 0 else None,
    pretrained_question_encoder_name_or_path=args.encoder_model_name_or_path if len(args.encoder_model_name_or_path) > 0 else None,
    pretrained_generator_tokenizer_name_or_path=args.generator_model_name_or_path,
    pretrained_generator_name_or_path=args.generator_model_name_or_path,
    **generator_config.to_diff_dict()
)

model = PreTrainedRagModel(config=config)
model.save_pretrained(args.checkpoint_path)
