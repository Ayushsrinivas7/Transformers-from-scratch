import torch 
import torch.nn as nn
import math 
import numpy 
class InputEmbeddings(nn.Module):
    
    def __init__(self , d_model : int , vocab_size : int ):
# note here dmodel is the dimension of the model ie the word embedding size here 512 
# vocab means total number of unique words in the entire data
#  we know we will be given data we need to tokenie them using the tokenizer 
# assign token ids by using the  the voabulary (list of all words)
# the words are coverted to vetors uing this tokens only 

        super().__init__()
        # we are intitalzing the parent class so that we can do use the parent methods
        self.d_model = d_model
        self.vocab_size = vocab_size
        #  now we are creating the embeddings for all the vocabulary 
        # ie if we have 100 words then we create the embeddings for 1 to 100 tokenids 
        #  so these embeddings are given to the corresponding words 
        self.embedding = nn.Embedding(vocab_size , d_model )
        # we are saying i have total num of wrds = vocab_size so form the embeddings 
        # each of size d_modal (512 for transformer )
        
    def forward(self  , x ):
        # here x ix the batch of tokens lets say x is of (batchsize , seq_len )
        # where we have the token ids inside . so now when we send this to forward 
        # we get the corresponding word embeddings in it so 
        # now (batchsize , seq_len , d-Model(512) ) 
        return self.embedding(x) * math.sqrt(self.d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        



# testing the input embeddings 
# Sample vocabulary
# ----------------------------------------------------------------------------
# vocab = ['hello', 'world', 'I', 'am', 'learning', 'PyTorch']
# word_to_idx = {word: i for i, word in enumerate(vocab)}

# # Create Input
# sentence = ['hello', 'world', 'PyTorch']
# input_indices = torch.tensor([[word_to_idx[word] for word in sentence]])

# # Model
# model = InputEmbeddings(d_model=4, vocab_size=len(vocab))
# output = model(input_indices)
# print("Embeddings for 'hello', 'world', 'PyTorch':\n", output)

# here the intention is to create the positional embeddings 
# we know we have to create the pos embed size same as wrd emebed 
# so that we can give the encode input wrd embed + pos embed 
# ----------------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self , d_model : int , seq_len : int , dropout : float ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len 
        # we are adding the dropout so that we can get the good accuracy  
        self.dropout = nn.Dropout(dropout)
        
        # create a shape (seq_len , d_modal )
        # as our inention is to create ebddings for all the words
        # in the sequence and the embeddings is given based on 
        # word position in sequence  
        pe = torch.zeros(self.seq_len , self.d_model )
        # lets create a vector of sequence length 
        # having the values from 0 to seq_len -1
        position = torch.arange( 0 , seq_len  , dtype = torch.float).unsqueeze(1)
        # .unsqueeze(1)
        # Purpose: Adds an extra dimension to the tensor at the specified position.
        # 1: Adds a new dimension at the second position (indexing starts from 0)
        #  now we have ( seq_len , 1 )
        # we want is sin(pos / (10000)^ (2i/d_modal)) and corresponding cos term 
        div_term = torch.exp( torch.arange(0 , d_model , 2).float() * (-math.log(10000) / d_model ))
        # this is of size ( 1 , d_modal/2 )
        pe[: , 0::2] = torch.sin(position * div_term )
        pe[: , 1:: 2] = torch.cos(position * div_term )
        # here we are ging the sin term for even terms and cos terms for the odd temrs 
        pe = pe.unsqueeze(0)
        # unsqueeze means it will add the dimension at the given location 
        # here it we are saying at 0 th pos ( 1 , seq , d_modal) will be the result 
        self.register_buffer('pe'  , pe )
        # note this is constant for every batch as every batch has same no of words
        # and same word embedding dim 
        #  the .register buffer will will sva e the positional embeddings
        # and it can be used any time we want and this will not be involved in the 
        # gradient  diesent as we know these are same for all
    def forward( self , x ):
        # here x is ( batch , seq_len , d_modal)
        x = x + ( self.pe[ : , : x.shape[1]  , : ]).requires_grad_(False)
        return self.dropout(x)

#-------------------------------------------------------------------------------------- 
# time_series = torch.randn( 3 , 20, 10)  # Batch size=16, seq_len=50, d_model=32
# encoder = PositionalEncoding(d_model = 10 , seq_len = 20 , dropout = 0.5)
# output = encoder.forward(time_series)
# print(output.shape) 
# --------------------------------------------------------------------------------------
class LayerNormalization(nn.Module):
    
    def __init__(self , features : int = 1 , eps : float = 10 **-6 ) -> None:
        super().__init__()
        self.features = features 
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        
    def forward(self , x ):
        mean = x.mean(dim = -1 , keepdim = True )
        std = x.std(dim = -1 , keepdim = True )
        return (self.alpha * ((x-mean )/ (std + self.eps) )) + self.bias 
    
        
# # Dummy input tensor
# x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# # Initialize LayerNormalization
# layer_norm = LayerNormalization(features=x.size(-1))

# # Apply layer normalization
# output = layer_norm(x)
# print("Input:\n", x)
# print("\nNormalized Output:\n", output)
# print("\n alpha value :\n", layer_norm.alpha )

# --------------------------------------------------------------------------------


#  the main need of the feed forward neural network is here adding the non linearity 

class FeedForwardBlock(nn.Module):
    
    def __init__(self , d_model : int , d_ff : int  , dropout : float ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model , d_ff ) 
        #  here we have the w1 and b1 
        self.linear_2 = nn.Linear(d_ff , d_model)
        # here we have the w2  and b2 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self , x ):
        # (batch , seq_len , d_model ) --> ( batch , seq_len ,  d_ff ) --> (batch , seq_len , d_model)
        return    self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
             

# # Dummy input tensor
# x = torch.randn(3, 10, 12)  # (batch_size, seq_len, d_model)

# # Initialize FeedForwardBlock
# ff_block = FeedForwardBlock(d_model=12, d_ff=8, dropout=0.1)

# # Apply the block
# output = ff_block(x)
# print("Output shape:", output.shape)

        
# --------------------------------------------------------------------------------------------

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self , d_model :int , h : int , dropout : float ) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        # we will make sure the d_model will be divided by h
        assert d_model % h == 0, 'd_model is not divisible by h'
        #  here h heads we will split the d_modal to (d_meal//h ) indidvidual components 
        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model , d_model  , bias = False  )
        self.w_k = nn.Linear(d_model , d_model , bias = False  )
        self.w_v = nn.Linear(d_model , d_model , bias = False )
        self.w_o = nn.Linear(d_model , d_model , bias = False )
        #  hey here w_query  , w_k , w_v isfor generation of q , k , v for the word embeding 
        
    @staticmethod 
    def attention(query , key , value  , mask , dropout : nn.Dropout ):
        
        d_k = query.shape[-1]
        # here the formula we need to implement is softmax( (q *kT )/sqrt(d_model) ) * v 
        attention_scores = (query @ key.transpose(-2 , -1))/math.sqrt(d_k)
        if mask is not None :
            attention_scores.masked_fill_(mask == 0 , float('-1e9')  )
            # here the intention of the mask is that when training at the decoder
            # we need need to ensure that the present is not depending on the future
            # because the decoder is auto regressive in inference and non auto regressiv at the training 
            #  ie we have to make sure that the present word dont dpend on the future word 
        attention_scores = attention_scores.softmax(dim = -1 )
        if dropout  is not None : 
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # (now we have the perfect attention scroes so lets make it use for values vector  )
        return ( attention_scores @ value )  , attention_scores
    
    def forward(self , q , k , v  , mask ):
        query  = self.w_q(q)
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model) this happend s for all 
        value  = self.w_v(v)
        key   = self.w_k(k)
        query = query.view( query.shape[0] , query.shape[1]  , self.h , self.d_k ).transpose(1 , 2)
        key = key.view( key.shape[0] , key.shape[1]  , self.h , self.d_k ).transpose(1 , 2)
        value = value.view( value.shape[0] , value.shape[1]  , self.h , self.d_k ).transpose(1 , 2)
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1 , 2 ).contiguous().view(x.shape[0], -1, self.h * self.d_k)
         # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
       
       
# def test_multihead_attention():
#     torch.manual_seed(0)
    
#     # Initialize block
#     d_model = 16  # Model dimensionality
#     h = 4         # Number of attention heads
#     seq_len = 8   # Sequence length
#     dropout = 0.1 # Dropout rate

#     mha = MultiHeadAttentionBlock(d_model, h, dropout)
#     q = torch.rand(2, seq_len, d_model)  # Query: (batch_size, seq_len, d_model)
#     k = torch.rand(2, seq_len, d_model)  # Key: (batch_size, seq_len, d_model)
#     v = torch.rand(2, seq_len, d_model)  # Value: (batch_size, seq_len, d_model)

#     # Optional mask (causal mask for decoder, for instance)
#     mask = torch.ones(2, 1, seq_len, seq_len)  # Mask: (batch_size, 1, seq_len, seq_len)

#     # Forward pass
#     output = mha(q, k, v, mask)
    
#     # Results
#     print("Output shape:", output.shape)  # Expected: (batch_size, seq_len, d_model)
#     print("Attention weights shape:", mha.attention_scores.shape)  # Expected: (batch_size, h, seq_len, seq_len)

# # Run the test
# test_multihead_attention()
         
                    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class ResidualConnection(nn.Module):
    
    def __init__(self , features:int , dropout:float ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # we will use the dropout layer for the prevention of overfitting
        self.norm = LayerNormalization(features)
        
    def forward(self , x , sublayer):
        
        # this is first method which is in the paper 
        # return  self.dropout(self.norm(x + sublayer(x)))
        # this is actual implementation as practical results look better for this 
        return x + self.dropout(sublayer(self.norm(x)))
        #  the reason is simple is the sublayer results are bad then we have the actual embed 
        #  note here the sublayer can be the self attention block or feed forward 
        # so thats why when sending the input tot he residual connection be send a labda fuction 
        # which take input x and processs the function iside lambda for self attention 
        # and does normal function like forward fr the feed forward 
        
   
# # this is testing the residual connection 
# # Define a simple sublayer for testing
# class SimpleSublayer(nn.Module):
#     def __init__(self, features: int):
#         super().__init__()
#         self.linear = nn.Linear(features, features)
    
#     def forward(self, x):
#         return self.linear(x)

# # Define test function
# def test_residual_connection():
#     features = 4
#     dropout = 0.1
#     batch_size = 2
#     seq_len = 5

#     # Instantiate ResidualConnection and SimpleSublayer
#     residual_connection = ResidualConnection(features, dropout)
#     sublayer = SimpleSublayer(features)

#     # Generate some random input tensor
#     x = torch.rand(batch_size, seq_len, features)

#     # Pass through ResidualConnection
#     output = residual_connection(x, sublayer)
    
#     print("Input Tensor:")
#     print(x)
#     print("\nOutput Tensor:")
#     print(output)

# # Run the test
# test_residual_connection()


# ---------------------------------------------------------------------------------------------------------------------------

# every encoder block has different ffedforward and multihead attention .
# but the layer norn poorcess is same 

class EncoderBlock(nn.Module):
    
    def __init__(self , features : int , self_attention_block : MultiHeadAttentionBlock , feed_forward_block : FeedForwardBlock  , dropout :  float )-> None:
        
        # """
        # Encoder block consisting of:
        # - A self-attention block with a residual connection.
        # - A feed-forward block with a residual connection.
        # - Dropout and LayerNorm for stability and regularization.
        # Args:
        #     features (int): Dimension of the model (d_model).
        #     self_attention_block (nn.Module): Multi-head self-attention block.
        #     feed_forward_block (nn.Module): Feed-forward network block.
        #     dropout (float): Dropout rate.
        # """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # as we know there are two residual connections one in multihead and other in feedforward 
        self.residual_connections = nn.ModuleList([ ResidualConnection(features , dropout ) for _ in range(2) ] )
        
    def forward(self, x, src_mask):
        # src_mask is used so that we will not consider the padding sequence 
        #   """
        # Forward pass of the EncoderBlock.
        # Args:
        #     x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        #     src_mask (Tensor): Source mask to prevent attending to certain positions.
        # Returns:
        #     Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        # """
        # Applying the first residual connection with the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # Three 'x's corresponding to query, key, and value inputs plus source mask

        # Applying the second residual connection with the feed-forward block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x # Output tensor after applying self-attention and feed-forward layers with residual connections.
        
        
            
# Test the EncoderBlock
# d_model = 8
# n_heads = 2
# seq_len = 5
# batch_size = 2

# # Initialize components
# self_attention_block = MultiHeadAttentionBlock(d_model = d_model , h = n_heads  , dropout =  0.3)
# feed_forward_block = FeedForwardBlock(d_model = d_model , d_ff=16  , dropout = 0.4)
# encoder_block = EncoderBlock(features=d_model, self_attention_block=self_attention_block, feed_forward_block=feed_forward_block, dropout=0.1)

# # Input tensor and mask
# x = torch.rand( batch_size, seq_len ,  d_model)  # Shape: (seq_len, batch_size, d_model)
# src_mask = None  # No mask for simplicity

# # Forward pass
# output = encoder_block(x, src_mask)
# print("Output Shape:", output.shape)  # Expected: (batch_size, seq_len , d_model)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# Building Encoder
# An Encoder can have several Encoder Blocks
class Encoder(nn.Module):

    # The Encoder takes in instances of 'EncoderBlock'
    
    #     """
    # Encoder consisting of multiple EncoderBlocks.
    # Applies normalization to the final output of all layers.

    # Args:
    #     features (int): Dimension of the model (d_model).
    #     layers (nn.ModuleList): List of EncoderBlock instances.
    # """
    def __init__(self, features : int ,  layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers # Storing the EncoderBlocks
        self.norm = LayerNormalization(features) # Layer for the normalization of the output of the encoder layers

    def forward(self, x, mask):
        
        #  """
        # Forward pass of the Encoder.
        # Args:
        #     x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        #     mask (Tensor): Source mask to prevent attention to certain positions.
        # Returns:
        #     Tensor: Normalized output tensor of shape (batch_size, seq_len, d_model).
        # """
        # Iterating over each EncoderBlock stored in self.layers
        for layer in self.layers:
            x = layer(x, mask) # Applying each EncoderBlock to the input tensor 'x'
        return self.norm(x) # Normalizing output
    
    
# # Testing the Encoder
# num_layers = 3
# d_model = 8
# n_heads = 2
# seq_len = 5
# batch_size = 3

# # Initialize components
# layers = nn.ModuleList([
#     EncoderBlock(
#         features=d_model,
#         self_attention_block=MultiHeadAttentionBlock(d_model=d_model, h=n_heads, dropout=0.1),
#         feed_forward_block=FeedForwardBlock(d_model=d_model, d_ff=16, dropout=0.1),
#         dropout=0.1
#     ) for _ in range(num_layers)
# ])

# encoder = Encoder(features=d_model, layers=layers)

# # Input tensor and mask
# x = torch.rand(batch_size, seq_len, d_model)  # Input tensor
# mask = None  # No mask for simplicity

# # Forward pass
# output = encoder(x, mask)
# print("Output Shape:", output.shape)  # Expected: (batch_size, seq_len, d_model)


# ------------------------------------------------------------------------------------------------------------------------------------------------------
   
   
# Building Decoder Block
class DecoderBlock(nn.Module):
    """
    A single block of the Transformer Decoder, which includes:
    - A self-attention block with residual connection.
    - A cross-attention block (attending to the encoder output) with residual connection.
    - A feed-forward block with residual connection.

    Args:
        features (int): Dimension of the model (d_model).
        self_attention_block (MultiHeadAttentionBlock): Multi-head self-attention block for target sequence.
        cross_attention_block (MultiHeadAttentionBlock): Multi-head cross-attention block for encoder-decoder interaction.
        feed_forward_block (FeedForwardBlock): Feed-forward network block.
        dropout (float): Dropout rate for residual connections.
    """

    # The DecoderBlock takes in two MultiHeadAttentionBlock. One is self-attention, while the other is cross-attention.
    # It also takes in the feed-forward block and the dropout rate
    def __init__(self, features : int ,   self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features , dropout) for _ in range(3)]) # List of three Residual Connections with dropout rate

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass for the DecoderBlock.

        Args:
            x (Tensor): Input tensor of shape (batch_size, tgt_seq_len, d_model).
            encoder_output (Tensor): Encoder output tensor of shape (batch_size, src_seq_len, d_model).
            src_mask (Tensor): Mask for source sequence (optional).
            tgt_mask (Tensor): Mask for target sequence (e.g., causal mask for self-attention).

        Returns:
            Tensor: Output tensor of shape (batch_size, tgt_seq_len, d_model).
        """
        # Self-Attention block with query, key, and value plus the target language mask
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))

        # The Cross-Attention block using two 'encoder_ouput's for key and value plus the source language mask. It also takes in 'x' for Decoder queries
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))

        # Feed-forward block with residual connections
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
    
    

# # Parameters
# d_model = 8
# n_heads = 2
# seq_len_tgt = 5
# seq_len_src = 6
# batch_size = 2
# dropout = 0.1

# # Create components
# self_attention_block = MultiHeadAttentionBlock(d_model=d_model, h=n_heads, dropout=dropout)
# cross_attention_block = MultiHeadAttentionBlock(d_model=d_model, h=n_heads, dropout=dropout)
# feed_forward_block = FeedForwardBlock(d_model=d_model, d_ff=16, dropout=dropout)

# # Create DecoderBlock
# decoder_block = DecoderBlock(
#     features=d_model,
#     self_attention_block=self_attention_block,
#     cross_attention_block=cross_attention_block,
#     feed_forward_block=feed_forward_block,
#     dropout=dropout
# )

# # Create input tensors
# x = torch.rand(batch_size, seq_len_tgt, d_model)  # Target sequence input
# encoder_output = torch.rand(batch_size, seq_len_src, d_model)  # Encoder output
# tgt_mask = torch.ones(seq_len_tgt, seq_len_tgt)  # Causal mask for target
# src_mask = None  # No mask for simplicity

# # Forward pass
# output = decoder_block(x, encoder_output, src_mask, tgt_mask)
# print("Output shape:", output.shape)  # Expected: (batch_size, seq_len_tgt, d_model)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Decoder(nn.Module):
    
    def __init__(self , features : int , layers : nn.ModuleList ) -> None : 
        """
        Decoder consists of multiple DecoderBlocks with layer normalization.
        
        Args:
            features (int): Feature size (d_model).
            layers (nn.ModuleList): List of DecoderBlock instances.
        """
        super().__init__()
        self.layers = layers
        self.norm  = LayerNormalization(features)
        
    def forward(self , x , encoder_output , src_mask , tgt_mask ):
        """
        Forward pass through the decoder.
        
        Args:
            x (Tensor): Input tensor (decoder input) of shape (batch_size, seq_len, d_model).
            encoder_output (Tensor): Encoder output of shape (batch_size, src_len, d_model).
            src_mask (Tensor): Source mask (optional) to prevent attending to padding tokens.
            tgt_mask (Tensor): Target mask (optional) to prevent attending to future tokens.
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        for layer in self.layers :
            x = layer(x , encoder_output  , src_mask  , tgt_mask )
            
        x = self.norm(x)
        return x
    
# # Test Decoder
# features = 8
# num_layers = 3
# seq_len = 5
# batch_size = 2

# # Create mock DecoderBlocks
# layers = nn.ModuleList([
#     DecoderBlock(
#         features =features, 
#         self_attention_block=MultiHeadAttentionBlock(features, h=2, dropout=0.1),
#         cross_attention_block=MultiHeadAttentionBlock(features, h=2, dropout=0.1),
#         feed_forward_block=FeedForwardBlock(d_model=features, d_ff=16, dropout=0.1),
#         dropout=0.1
#     ) 
#     for _ in range(num_layers)
# ])

# # Create Decoder
# decoder = Decoder(features = features, layers=layers)

# # Inputs
# x = torch.rand(batch_size, seq_len, features)  # Decoder input
# encoder_output = torch.rand(batch_size, seq_len, features)  # Encoder output
# src_mask = None  # Source mask
# tgt_mask = None  # Target mask

# # Forward pass
# output = decoder(x, encoder_output, src_mask, tgt_mask)
# print("Output Shape:", output.shape)  # Expected: (batch_size, seq_len, features)

    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
  

# Buiding Linear Layer
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None: # Model dimension and the size of the output vocabulary
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) # Linear layer for projecting the feature space of 'd_model' to the output space of 'vocab_size'
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim = -1) # Applying the log Softmax function to the output
    
    
              
# # Define dimensions
# d_model = 8
# vocab_size = 10
# seq_len = 5
# batch_size = 2

# # Create the projection layer
# projection_layer = ProjectionLayer(d_model, vocab_size)

# # Input tensor (random embeddings)
# x = torch.rand(batch_size, seq_len, d_model)

# # Forward pass
# output = projection_layer(x)

# # Print the shape of the output
# print("Input Shape:", x.shape)        # Expected: (batch_size, seq_len, d_model)
# print("Output Shape:", output.shape)  # Expected: (batch_size, seq_len, vocab_size)


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, 
                 encoder: nn.Module, 
                 decoder: nn.Module, 
                 src_embed: nn.Module, 
                 tgt_embed: nn.Module, 
                 src_pos: nn.Module, 
                 tgt_pos: nn.Module, 
                 projection_layer: nn.Module) -> None:
        """
        Transformer model consisting of an Encoder, Decoder, input embeddings, positional encodings, and a projection layer.

        Args:
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
            src_embed (nn.Module): Source input embedding module.
            tgt_embed (nn.Module): Target input embedding module.
            src_pos (nn.Module): Positional encoding for source.
            tgt_pos (nn.Module): Positional encoding for target.
            projection_layer (nn.Module): Projection layer for mapping to vocabulary size.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes the source sequence.
        Args:
            src (Tensor): Source sequence of shape (batch_size, src_seq_len).
            src_mask (Tensor): Source mask of shape (batch_size, 1, src_seq_len).
        Returns:
            Tensor: Encoded source of shape (batch_size, src_seq_len, d_model).
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, 
               encoder_output: torch.Tensor, 
               src_mask: torch.Tensor, 
               tgt: torch.Tensor, 
               tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Decodes the target sequence using encoded source information.
        Args:
            encoder_output (Tensor): Output from the encoder of shape (batch_size, src_seq_len, d_model).
            src_mask (Tensor): Source mask of shape (batch_size, 1, src_seq_len).
            tgt (Tensor): Target sequence of shape (batch_size, tgt_seq_len).
            tgt_mask (Tensor): Target mask of shape (batch_size, tgt_seq_len, tgt_seq_len).
        Returns:
            Tensor: Decoded target of shape (batch_size, tgt_seq_len, d_model).
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the model output to the vocabulary space.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Tensor: Projected tensor of shape (batch_size, seq_len, vocab_size).
        """
        return self.projection_layer(x)

        
# # Define dummy components
# d_model = 8
# vocab_size = 10
# seq_len = 5
# batch_size = 2

# # Dummy encoder, decoder, embeddings, positional encodings, and projection layer
# encoder = Encoder(features=d_model, layers=nn.ModuleList([EncoderBlock(d_model, MultiHeadAttentionBlock(d_model, 2, 0.1), FeedForwardBlock(d_model, 16, 0.1), 0.1)]))
# decoder = Decoder(features=d_model, layers=nn.ModuleList([DecoderBlock(d_model, MultiHeadAttentionBlock(d_model, 2, 0.1), MultiHeadAttentionBlock(d_model, 2, 0.1), FeedForwardBlock(d_model, 16, 0.1), 0.1)]))
# src_embed = nn.Embedding(20, d_model)
# tgt_embed = nn.Embedding(20, d_model)
# src_pos = nn.Identity()  # For simplicity
# tgt_pos = nn.Identity()  # For simplicity
# projection_layer = ProjectionLayer(d_model, vocab_size)

# # Initialize Transformer
# transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

# # Dummy inputs
# src = torch.randint(0, 20, (batch_size, seq_len))
# tgt = torch.randint(0, 20, (batch_size, seq_len))
# src_mask = torch.ones(batch_size, 1, seq_len)
# tgt_mask = torch.ones(batch_size, seq_len, seq_len)

# # Forward pass
# encoded = transformer.encode(src, src_mask)
# decoded = transformer.decode(encoded, src_mask, tgt, tgt_mask)
# output = transformer.project(decoded)

# print("Encoded Shape:", encoded.shape)  # Expected: (batch_size, seq_len, d_model)
# print("Decoded Shape:", decoded.shape)  # Expected: (batch_size, seq_len, d_model)
# print("Output Shape:", output.shape)    # Expected: (batch_size, seq_len, vocab_size)



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 
# Building & Initializing Transformer

# Definin function and its parameter, including model dimension, number of encoder and decoder stacks, heads, etc.
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:

    # Creating Embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size) # Source language (Source Vocabulary to 512-dimensional vectors)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size) # Target language (Target Vocabulary to 512-dimensional vectors)

    # Creating Positional Encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout) # Positional encoding for the source language embeddings
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout) # Positional encoding for the target language embeddings

    # Creating EncoderBlocks
    encoder_blocks = [] # Initial list of empty EncoderBlocks
    for _ in range(N): # Iterating 'N' times to create 'N' EncoderBlocks (N = 6)
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout) # Self-Attention
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout) # FeedForward

        # Combine layers into an EncoderBlock
        encoder_block = EncoderBlock(d_model , encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block) # Appending EncoderBlock to the list of EncoderBlocks

    # Creating DecoderBlocks
    decoder_blocks = [] # Initial list of empty DecoderBlocks
    for _ in range(N): # Iterating 'N' times to create 'N' DecoderBlocks (N = 6)
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout) # Self-Attention
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout) # Cross-Attention
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout) # FeedForward

        # Combining layers into a DecoderBlock
        decoder_block = DecoderBlock(d_model , decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block) # Appending DecoderBlock to the list of DecoderBlocks

    # Creating the Encoder and Decoder by using the EncoderBlocks and DecoderBlocks lists
    encoder = Encoder(d_model ,nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model , nn.ModuleList(decoder_blocks))

    # Creating projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size) # Map the output of Decoder to the Target Vocabulary Space

    # Creating the transformer by combining everything above
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer # Assembled and initialized Transformer. Ready to be trained and validated!
    
        
        
        

# # Example Testing Function
# def test_transformer():
#     # Example configuration
#     src_vocab_size = 10000
#     tgt_vocab_size = 10000
#     src_seq_len = 50
#     tgt_seq_len = 50
#     batch_size = 16
#     d_model = 512

#     # Build the Transformer model
#     transformer = build_transformer(
#         src_vocab_size=src_vocab_size,
#         tgt_vocab_size=tgt_vocab_size,
#         src_seq_len=src_seq_len,
#         tgt_seq_len=tgt_seq_len,
#         d_model=d_model,
#         N=6,  # Number of Encoder/Decoder layers
#         h=8,  # Number of attention heads
#         dropout=0.1,
#         d_ff=2048  # Feed-forward layer dimension
#     )

#     # Input tensors for testing
#     src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))  # Random source tokens
#     tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))  # Random target tokens

#     # Masks (optional, use None for simplicity here)
#     src_mask = None
#     tgt_mask = None

#     # Forward pass through the Transformer
#     encoder_output = transformer.encode(src, src_mask)
#     decoder_output = transformer.decode(encoder_output, src_mask, tgt, tgt_mask)
#     final_output = transformer.project(decoder_output)

#     # Print Shapes to Validate
#     print("Source Input Shape:", src.shape)
#     print("Target Input Shape:", tgt.shape)
#     print("Encoder Output Shape:", encoder_output.shape)
#     print("Decoder Output Shape:", decoder_output.shape)
#     print("Final Output Shape:", final_output.shape)  # Expected: (batch_size, tgt_seq_len, tgt_vocab_size)

#     # Ensure the final output has the correct dimensions
#     assert final_output.shape == (batch_size, tgt_seq_len, tgt_vocab_size), "Output dimensions are incorrect!"

# # Run the test
# test_transformer()

   
   
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
           
        
        
