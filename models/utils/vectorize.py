class VectorizeTexts(object):
    def __init__(self, config):
        """Preparing text features."""
        
        self._mode_bin = config.get('mode_bin', True)
        self._mode_idf = config.get('mode_idf', True)
        self._mode_tf = config.get('mode_tf', True)
        self._mode_tfidf = config.get('mode_tfidf', True)
        self._mode_scale = config.get('mode_scale', True)
        
    def _get_bin(self, result):
        """Get binary vectors"""
        
        result = (result > 0).astype('float32')
        
        return result
    
    def _get_tf(self, result):
        """Get term frequency."""
        
        result = result.tocsr()
        result = result.multiply(1 / result.sum(1))
        
        return result
    
    def _get_idf(self, result):
        """Get term frequency and inverse document frequency."""
        
        result = result.tocsr()
        result = result.multiply(1 / result.sum(1)) 
        result = result.multiply(1 / word2freq) 
        
        return result 
    
    def _get_scale(self, result):
        """Standardize Tfidf dataset."""
        
        result = result.tocsc()
        result -= result.min()
        result /= (result.max() + 1e-6)
        
        return result
    
    def transform(self, tokenized_texts, word2id, word2freq):
        
        result = scipy.sparse.dok_matrix((len(tokenized_texts), len(word2id)), dtype='float32')
        for text_i, text in enumerate(tokenized_texts):
            for token in text:
                if token in word2id:
                    result[text_i, word2id[token]] += 1
        
        
        if self._mode_bin:
            result = self._get_bin(result)
        
        if self._mode_idf:
            result = self._get_idf(result)
            
        if self._mode_tf:
            result = self._get_tf(result)
            
        if self._mode_tfidf:
            result = self._get_tfidf(result)
            
        if self._mode_scale:
            result = self._get_scale(result)
            
        return result.tocsr()
            