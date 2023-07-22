
import torch
from models.predictor import MatchingClassifier
import torch.nn as nn

class BertClassifier(nn.Module):
    def __init__(self, encoder,num_labels=18):
        super(BertClassifier, self).__init__()
        self.bert = encoder
        self.num_labels = num_labels
        self.classifier = nn.Linear(768, num_labels,bias=True)
        init_transitions_param = torch.zeros(10, 10)
        self.crf_pos_transitions_model = nn.Parameter(init_transitions_param)

    def forward(self, inputs, lengths, input_mask):
        outputs = self.bert.forward(input_ids=inputs['input_tf']['input_ids'],
                                 attention_mask=inputs['input_tf']['attention_mask'],
                                 token_type_ids=inputs['input_tf']['segment_ids'])
        pooled_output = outputs[1]
        sequence_output = outputs[0]

        chosen_encoder_hiddens = sequence_output.reshape(-1, 768).index_select(0, inputs['input_tf']['selects'])
        select_support_pos_embeds = torch.zeros(
            inputs['input_tf']['batch_size'] * inputs['input_tf']['max_word_length'], 768,
            device=sequence_output.device)
        select_support_pos_embeds = select_support_pos_embeds.index_copy_(0, inputs['input_tf']['copies'],
                                                                          chosen_encoder_hiddens).view(
            inputs['input_tf']['batch_size'], inputs['input_tf']['max_word_length'], -1)

        logits = self.classifier(select_support_pos_embeds)
        return logits,select_support_pos_embeds
    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))
    def load_model(self, load_dir):
        self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))



class FusionNet(nn.Module):
    def __init__(self,config,matching_similarity_function):
        super(FusionNet, self).__init__()
        self.base_fusion = nn.Linear(768*2, 768,bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.base_weight = nn.Linear(768, 768, bias=True)
        # self.novel_fusion = nn.Linear(768*2, 768, bias=True)
        # self.train_fusion = nn.Linear(768*2,768,bias=True)
        # self.tanh = nn.Tanh()
        # self.novel_weight = nn.Linear(768, 768, bias=True)
        # self.train_weight = nn.Linear(768,768,bias=True)
        # self.slot_tagger = MatchingClassifier(config, matching_similarity_function=matching_similarity_function)



    def forward(self, slot_tag_embeds_base,slot_tag_embeds_novel,slot_tag_embeds_pos_abs, slot_tag_embeds_abs_novel,mid_batch,similar_weight
                ):
        z = torch.cat((slot_tag_embeds_base, slot_tag_embeds_novel),axis=-1)
        # p = torch.cat((slot_tag_embeds_pos_abs, slot_tag_embeds_abs_novel),axis=-1)

        deep_z = self.relu(self.base_fusion(z))
        base_wz = self.sigmoid(self.base_weight(deep_z))

        # deep_p = self.relu(self.train_fusion(p))
        # base_wp = self.sigmoid(self.train_weight(deep_p))

        new_slot_tag_embeds = 1 * (base_wz * slot_tag_embeds_base + (1 - base_wz) * slot_tag_embeds_novel)
        # new_slot_tag_embeds = 0.5 * slot_tag_embeds_novel + 0.5 * slot_tag_embeds_abs_base
        slot_tag_embeds_abs = 0.5 * slot_tag_embeds_pos_abs + 0.5 * slot_tag_embeds_abs_novel
        return new_slot_tag_embeds,slot_tag_embeds_abs

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))
    def load_model(self, load_dir):
        self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))