INFERRING FUNCTION NAMES IN STRIPPED BINARIES WITH CODEBERT

Naming pieces of assembly code plays a crucial role in malware analysis, which is usually carried
by a reverse engineer who has a high understanding of assembly codes and malware behaviors.
Recently, several pre-trained deep learning models for natural language and programming lan-
guages (e.g, Roberta, CodeBERT, and GraphCodeBert) have shown promising results in main-
stream Natural Language Processing (NLP) tasks. Taking advantage of the powerful capacity of
those pre-trained models, this report investigates the feasibility of considering function labeling as
a text generation problem. A CodeBERT-based Encoder-Decoder is used to learn the relation-
ship between function labels and some intermediate presentations of binaries to predict names in
stripped binary. Two solutions for the intermediate presentation of binary, decompiled code, and
assembly with VEX instructions, are tested. The analysis performed over open-source C binaries
from the Linux distribution and real-world unstripped malware from VirusShare show good F1-
score and BLEU score comparing to existing works. Moreover, the model trained on unstripped
malware dataset is capable of inferring reasonable function names when testing on stripped Mirai
binary
