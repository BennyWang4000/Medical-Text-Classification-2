class TextPreprocessor(object):
    def __init__(self, config):
        """Preparing text features."""

        self._del_orig_col = config.get('del_orig_col', True)
        self._mode_stemming = config.get('mode_stemming', True)
        self._mode_norm = config.get('mode_norm', True)
        self._mode_remove_stops = config.get('mode_remove_stops', True)
        self._mode_drop_long_words = config.get('mode_drop_long_words', True)
        self._mode_drop_short_words = config.get('mode_drop_short_words', True)
        self._min_len_word = config.get('min_len_word', 3)
        self._max_len_word = config.get('max_len_word', 17)
        self._max_size_vocab = config.get('max_size_vocab', 100000)
        self._max_doc_freq = config.get('max_doc_freq', 0.8) 
        self._min_count = config.get('min_count', 5)
        self._pad_word = config.get('pad_word', None)

    def _clean_text(self, input_text):
        """Delete special symbols."""

        input_text = input_text.str.lower()
        input_text = input_text.str.replace(r'[^a-z ]+', ' ')
        input_text = input_text.str.replace(r' +', ' ')
        input_text = input_text.str.replace(r'^ ', '')
        input_text = input_text.str.replace(r' $', '')

        return input_text


    def _text_normalization_en(self, input_text):
        '''Normalization of english text'''

        return ' '.join([lemmatizer.lemmatize(item) for item in input_text.split(' ')])


    def _remove_stops_en(self, input_text):
        '''Delete english stop-words'''

        return ' '.join([w for w in input_text.split() if not w in stop_words_en])


    def _stemming_en(self, input_text):
        '''Stemming of english text'''

        return ' '.join([stemmer_en.stem(item) for item in input_text.split(' ')])


    def _drop_long_words(self, input_text):
        """Delete long words"""
        return ' '.join([item for item in input_text.split(' ') if len(item) < self._max_len_word])


    def _drop_short_words(self, input_text):
        """Delete short words"""

        return ' '.join([item for item in input_text.split(' ') if len(item) > self._min_len_word])
    
    
    def _build_vocabulary(self, tokenized_texts):
        """Build vocabulary"""
        
        word_counts = collections.defaultdict(int)
        doc_n = 0

        for txt in tokenized_texts:
            doc_n += 1
            unique_text_tokens = set(txt)
            for token in unique_text_tokens:
                word_counts[token] += 1
                
        word_counts = {word: cnt for word, cnt in word_counts.items()
                       if cnt >= self._min_count and cnt / doc_n <= self._max_doc_freq}
        
        sorted_word_counts = sorted(word_counts.items(),
                                    reverse=True,
                                    key=lambda pair: pair[1])
        
        if self._pad_word is not None:
            sorted_word_counts = [(pad_word, 0)] + sorted_word_counts
            
        if len(word_counts) > self._max_size_vocab:
            sorted_word_counts = sorted_word_counts[:self._max_size_vocab]
            
        word2id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}
        word2freq = np.array([cnt / doc_n for _, cnt in sorted_word_counts], dtype='float32')

        return word2id, word2freq

    def transform(self, df):        
        
        columns_names = df.select_dtypes(include='object').columns
        df[columns_names] = df[columns_names].astype('str')
        
        for i in df.index:
            df.loc[i, 'union_text'] = ' '.join(df.loc[i, columns_names])
            
        if self._del_orig_col:
            df = df.drop(columns_names, 1)
            
        df['union_text'] = self._clean_text(df['union_text'])
        
        if self._mode_norm:
            df['union_text'] = df['union_text'].apply(self._text_normalization_en, 1)
            
        if self._mode_remove_stops:
            df['union_text'] = df['union_text'].apply(self._remove_stops_en, 1)
            
        if self._mode_stemming:
            df['union_text'] = df['union_text'].apply(self._stemming_en)
            
        if self._mode_drop_long_words:
            df['union_text'] = df['union_text'].apply(self._drop_long_words, 1)
            
        if self._mode_drop_short_words:
            df['union_text'] = df['union_text'].apply(self._drop_short_words, 1)
            
        df.loc[(df.union_text == ''), ('union_text')] = 'EMPT'
        
        tokenized_texts = [[word for word in text.split(' ')] for text in df.union_text]
        word2id, word2freq = self._build_vocabulary(tokenized_texts)

        return tokenized_texts, word2id, word2freq