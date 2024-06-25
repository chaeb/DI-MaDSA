SPECIAL_TOKENS = ["<s>", "</s>", "<usr>", "<sys>", "<pad>","<emo>", "<pos>", "<neg>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>', 'additional_special_tokens': ['<usr>', '<sys>', '<emo>', '<pos>', '<neg>']}
MODEL_INPUTS = ["input_ids", "labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "labels", "token_type_ids"]
PHQ_TOKENS = ['<depressed>', '<interest>', '<sleep>', '<appetite>', '<failure>', '<concentrating>', '<tired>', '<moving>']