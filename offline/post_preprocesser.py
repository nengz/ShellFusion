import warnings
import re

from nltk import sent_tokenize, PorterStemmer, word_tokenize
from nltk.corpus import stopwords

warnings.filterwarnings(action='ignore', category=UserWarning, module='bs4')

from bs4 import BeautifulSoup

long_code_pstrs = [
    r'<pre.*?>\s*?<code>(.*?\n)*?.*?</code>\s*</pre>',
    r'<pre.*?>\s*?<code>(.*?\n)*?.*?</pre>\s*</code>',
    r'<code>\s*?<pre.*?>(.*?\n)*?.*?</pre>\s*</code>',
    r'<code>\s*?<pre.*?>(.*?\n)*?.*?</code>\s*</pre>',
    r'<pre.*?>(.*?\n)*?.*?</pre>'
]

long_code_patterns = list()
for reg_str in long_code_pstrs:
    long_code_patterns.append(re.compile(reg_str, re.IGNORECASE))

short_code_pattern = re.compile(r'<code>(.*?\n)*?.*?</code>', re.IGNORECASE)
link_pattern = re.compile(r'<a .*?href=.*?>', re.IGNORECASE)
blockquote_pattern = re.compile(r'<blockquote>(.*?\n)*?.*?</blockquote>', re.IGNORECASE)

porter_stemmer = PorterStemmer()
punc_pattern = re.compile(r'[()/:\\\'\",?`\[\]{}!=;|@_]')
token_stem_dict = {}

eng_stopwords = set()
for sw in stopwords.words('english'):
    eng_stopwords.add(sw)


def cleanText(text):
    """
    Clean a text, including the blockquotes, long code snippets, and links.
    """
    items = re.finditer(blockquote_pattern, text)
    for item in items:
        s = item.group()
        text = text.replace(s, '')

    for lcp in long_code_patterns:
        items = re.finditer(lcp, text)
        for item in items:
            fcs = item.group()
            text = text.replace(fcs, ' ', 1)

    items = re.finditer(link_pattern, text)
    for item in items:
        link = item.group()
        i = text.find(link) + len(link)
        entity = text[i:text.find('</a>', i)]
        text = text.replace(link + entity + '</a>', entity, 1)

    if text != '':
        text = cleanHtmlTags(text)
        if text is not None:
            return ' '.join(text.split())
    return ''


def cleanHtmlTags(s):
    """
    Clean html tags in a string.
    """
    try:
        soup = BeautifulSoup(s, 'html.parser', from_encoding='utf-8')
        return soup.get_text()
    except Exception as e:
        print('***** ERROR in cleanHtmlTags():', e, '->', s)
        return None


def preprocessStr(s, s_type):
    """
    Preprocess a string (a body or a title or a query).
    """
    if s_type == '1':
        s = cleanText(s)
    s = re.sub('[^a-zA-Z0-9-_+:.?!,;]', ' ', s.replace('\n', ' '))
    s = re.sub(' +', ' ', s.lower())
    psens = []

    for sen in sent_tokenize(s):

        words = []
        sen = re.sub('[.?!,;]', ' ', sen)
        for token in word_tokenize(sen):
            if token in eng_stopwords or len(token) > 100:
                continue
            if token in token_stem_dict:
                words.append(token_stem_dict[token])
            else:
                try:
                    stem = porter_stemmer.stem(token)
                    if stem is not None and stem != '':
                        token_stem_dict[token] = stem
                        words.append(stem)
                except Exception as e:
                    print('***** Error in StemToken *****', e, ':', token)

        if len(words) > 0:
            if s_type != '1' or len(words) > 3:
                psens.append(' '.join(words))

    return '\n'.join(psens)


def removeStopWords(s):
    """
    Remove stop words in a string.
    """
    return ' '.join([ t for t in s.split() if t not in eng_stopwords ])
