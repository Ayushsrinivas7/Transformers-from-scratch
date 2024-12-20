import torch 
import torch.nn as nn 
from torch.utils.data import Dataset  , DataLoader 

# in pytorch  it is standard that if any one ask you to create 
# a dataset class then 
# create a class and inherit the Dataset which is in torch.utils.data 
# then give 3 properties 1) __init__ 2) __len__ 3) __getitem__
# all the maniculations and other things have to be done here 
 
class BilingualDataset(Dataset):
    # inherit the dataset class 
    
    def __init__(self , ds , tokenizer_src , tokenizer_tgt , src_lang , tgt_lang , seq_len ):
        super().__init__()
        
        self.seq_len = seq_len
        self.ds = ds 
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # note as the tokenizer special tokens are smae for src_tokenizer and tgt_tokenizer we h=will have the 
        # same id for the special toekns 
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")] , dtype = torch.int64 )
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]") ]  , dtype = torch.int64 )
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]") ]  , dtype = torch.int64 )
        
    def __len__(self):
            return len(self.ds)
        # we will return the length for the ds 
        
    def __getitem__(self , idx ):
            # here the traget is designing how the data smaple
            # should be and evaluate it 
            
            # step1 extarct the texts 
            src_target_pair = self.ds[idx]
            src_text = src_target_pair['translation'][self.src_lang]
            tgt_text = src_target_pair['translation'][self.tgt_lang]
            
            # step2 convrt the text to tokenids using the tokenizer 
            enc_input_tokens = self.tokenizer_src.encode(src_text).ids
            dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids 
            # dont foget to give .ids  if not ypu will get the tokens only
            
            
            # step3 preprocessing the things ie adding the things like 
            # sos , eos , padding and other things 
            
            # step3.1 lets calulate the number of pad we can add
            # as we know for the encode senteve we can add sos , eos 
            # but for the decoder input we add only sos 
            # and for the labels we will add the eos 
            #  and after that in order tot make all the sentnces of same size we will add the tokens 
            
            enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) -2  
            #  2 as we add sos and eos  
            dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) -1 
            # 1 as we add eos 
            if enc_num_padding_tokens <0 or dec_num_padding_tokens < 0 :
                raise ValueError("sentence is too long ie exceeded the sequence length ")
            
            # now lets add the things 
            
            # encoder input new is 
            encoder_input = torch.cat(
                 [
                     self.sos_token , 
                     torch.tensor(enc_input_tokens, dtype = torch.int64),
                     self.eos_token,
                     torch.tensor([self.pad_token] * enc_num_padding_tokens , dtype = torch.int64 )  
                  ], 
                 dim =0 
            )
            
            # new decoder output is :
            decoder_input = torch.cat(
                 [
                     self.sos_token , 
                     torch.tensor(dec_input_tokens, dtype = torch.int64),
                     torch.tensor([self.pad_token] * dec_num_padding_tokens , dtype = torch.int64 ),
                  ] , 
                 dim =0 
            )
            
            
            # labels which are used for the loss caluclation and backwardpass
            
            label = torch.cat(
                 [
                     
                     torch.tensor(dec_input_tokens, dtype = torch.int64),
                     self.eos_token,
                     torch.tensor([self.pad_token] * dec_num_padding_tokens , dtype = torch.int64 ),
                  ] , 
                 dim =0
            )
            
            
            assert encoder_input.size(0) == self.seq_len
            assert decoder_input.size(0) == self.seq_len
            assert label.size(0) == self.seq_len
            
            return{
                "encoder_input": encoder_input,  # (seq_len)
                "decoder_input": decoder_input,  # (seq_len)
                "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
                # this i made change as per chat gpt as it told both should be same size
                "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
                "label": label,  # (seq_len)
                "src_text": src_text,
                "tgt_text": tgt_text,  
            }
             
            
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask.type(torch.int64) == 0         
            
            
          
         
    