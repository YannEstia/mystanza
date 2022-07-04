import stanza
from modules import pipelinebuilder as pb

#stanza.download('en')

# nlp = stanza.Pipeline('en')

# doc = nlp("Barack Obama was born in Hawaii.")

# for sent in doc.sentences:
#     for word in sent.words:
#         print("text: ",word.text," - lemma : ", word.lemma," - POS : ", word.pos)

# print("~~~~~~")

# for sent in doc.sentences:
#     print("@@@@@@@@@@@ entities : ", sent.ents)
#     print("############ dependencies : ", sent.dependencies)

step1 = pb.my_stanza_Tokenizer("Barack Obama was born in Hawaii.")
print('\n*** end of step 1 ***\n')

step2 = pb.my_stanza_MWTExpansion("Barack Obama was born in Hawaii.")
print('\n*** end of step 2 ***\n')

step3 = pb.my_stanza_POSTagger("Barack Obama was born in Hawaii.")
print('\n*** end of step 3 ***\n')

step4 = pb.my_stanza_Lemmatizer("Barack Obama was born in Hawaii.")
print('\n*** end of step 4 ***\n')

step5 = pb.my_stanza_DepPars("Barack Obama was born in Hawaii. After graduating from Columbia University in 1983, he worked as a community organizer in Chicago. In 1988, he enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review.")
print('\n*** end of step 5 ***\n')

stepNER = pb.my_stanza_NER("Barack Obama was born in Hawaii. After graduating from Columbia University in 1983, he worked as a community organizer in Chicago. In 1988, he enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review.")