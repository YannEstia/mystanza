from pyexpat import model
import stanza

def my_stanza_Tokenizer(input_text):
    tokenize_processor = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'})
    doc = tokenize_processor(input_text)
    for i, sent in enumerate(doc.sentences):
        print(f'===== Sentence {i+1} tokens ======')
        print(*[f'id: {token.id}\ttext: {token.text}' for token in sent.tokens], sep='\n')
    

def my_stanza_MWTExpansion(input_text):
    mwt_processor = stanza.Pipeline(lang='en', processors='tokenize,mwt')
    doc = mwt_processor(input_text)
    for token in doc.sentences[0].tokens:
        print(f'token: {token.text}\twords: {", ".join([word.text for word in token.words])}')

def my_stanza_POSTagger(input_text):
    pos_processor = stanza.Pipeline(lang='en', processors='tokenize, mwt, pos')
    doc = pos_processor(input_text)
    print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')


def my_stanza_Lemmatizer(input_text):
    lemma_processor = stanza.Pipeline(lang='en', processors='tokenize, mwt, pos, lemma')
    doc = lemma_processor(input_text)
    print(*[f'word: {word.text+" "}\tlemma: {word.lemma}' for sent in doc.sentences for word in sent.words], sep='\n')

def my_stanza_DepPars(input_text):
    deppars_processor = stanza.Pipeline(lang='en', processors='tokenize, mwt, pos, lemma, depparse', depparse_batch_size = 1000, use_gpu = False)
    doc = deppars_processor(input_text)
    print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')

def my_stanza_NER(input_text):
    NER_processor = stanza.Pipeline(lang='en', processors='tokenize, ner', use_gpu=False)
    doc = NER_processor(input_text)
    print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc.ents], sep='\n')