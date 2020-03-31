# Models

### 简单二元模型

- process every sentence

  - calculate $P(w_i | w_{i-1})$ for every word
  - calculate the frequency of every word, $P(w_i)$

- Given a sequence of pinyin, i.e. O, need to find X, s.t. $P(X|O)$ is max
  $$
  P(X|O) = P(w_i|w_{i-1}O))*P(w_{i-2}|w_{i-1}O)...P(w_1|O)
  $$

- Use viterbi to find the best X



# Framework

- Should allow different methods to compute the model, i.e. models should have the same interfaces
- Modules:
  
  - Model interface
    - train(filename: str)
    - translate(input: str) -> str
  
  - Pipeline
    - Call model.train on data file
    - Call translate for all input strs
    - Calculate error rate