# Deep Learning Papers Classés par Difficulté
---

- https://colab.research.google.com/drive/1PVHupMXgJwfAVrTWhJQcafxUcqj8IqkD?usp=sharing
- https://github.com/zszazi/Deep-learning-in-cloud
- data : https://mrinalxdev.github.io/mrinalxblogs/blogs/dsa-practical.html
- https://www.youtube.com/playlist_?list=PL1v8zpldgH3pXDttKKp8mlVKDitxsYDAp
- - https://onedrive.live.com/?cid=ae6716deea45ca05&id=AE6716DEEA45CA05%21sdde779b339e047aaa7300c92771306be&resid=AE6716DEEA45CA05%21sdde779b339e047aaa7300c92771306be&ithint=folder&e=3VGYop&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy9hZTY3MTZkZWVhNDVjYTA1L0VyTjU1OTNnT2FwSHB6QU1rbmNUQnI0QmRxNHJGSi0weENuQ2tRX3VoZlJzN3c%5FZT0zVkdZb3A&v=validatepermission
Outils: 
- https://scholarqa.allen.ai/chat
- https://www.semanticscholar.org
### **Niveau 1 : Fondations (Facile)**
1. **Perceptron** (1958) - *Rosenblatt*  
   ["The Perceptron: A Probabilistic Model..."](https://psycnet.apa.org/record/1959-09865-001)
    - **Concepts** : Neurone artificiel, fonction d'activation.
    - **Difficulté** : ★☆☆☆☆ (Pas de calcul tensoriel, logique booléenne).

2. **Backpropagation** (1986) - *Rumelhart, Hinton, Williams*  
   ["Learning Representations by Back-Propagating Errors"](https://www.nature.com/articles/323533a0)
    - **Concepts** : Descente de gradient, chaînage des dérivées.
    - **Difficulté** : ★★☆☆☆ (Calcul différentiel de base).

---

### **Niveau 2 : Architectures Classiques (Intermédiaire)**
3. **LeNet-5** (1998) - *LeCun et al.*  
   ["Gradient-Based Learning Applied to Document Recognition"](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
    - **Concepts** : Convolutions, pooling, MNIST.
    - **Difficulté** : ★★☆☆☆ (Code simple en PyTorch/TensorFlow).

4. **AlexNet** (2012) - *Krizhevsky et al.*  
   ["ImageNet Classification with Deep Convolutional..."](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
    - **Concepts** : ReLU, Dropout, GPUs.
    - **Difficulté** : ★★☆☆☆ (Architecture CNN évoluée).

5. **Word2Vec** (2013) - *Mikolov et al.*  
   ["Efficient Estimation of Word Representations..."](https://arxiv.org/abs/1301.3781)
    - **Concepts** : Embeddings, Skip-gram/CBOW.
    - https://jalammar.github.io/illustrated-word2vec/
    - https://jalammar.github.io/visual-numpy/
    - **Difficulté** : ★★☆☆☆ (Algorithme d'optimisation simple).

---

### **Niveau 3 : Optimisation et Régularisation (Intermédiaire+)**
6. **Dropout** (2014) - *Srivastava et al.*  
   ["Dropout: A Simple Way to Prevent Neural Networks..."](https://jmlr.org/papers/v15/srivastava14a.html)
    - **Concepts** : Régularisation, entraînement stochastique.
    - **Difficulté** : ★★☆☆☆ (Intuition probabiliste).

7. **Batch Normalization** (2015) - *Ioffe, Szegedy*  
   ["Batch Normalization..."](https://arxiv.org/abs/1502.03167)
    - **Concepts** : Covariate shift, normalisation.
    - **Difficulté** : ★★★☆☆ (Statistiques par batch).

8. **Adam Optimizer** (2014) - *Kingma, Ba*  
   ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980)
    - **Concepts** : Moments adaptatifs, taux d'apprentissage.
    - **Difficulté** : ★★★☆☆ (Momentum + RMSProp combinés).

---

### **Niveau 4 : Architectures Profondes (Avancé)**
9. **ResNet** (2015) - *He et al.*  
   ["Deep Residual Learning..."](https://arxiv.org/abs/1512.03385)
    - **Concepts** : Residual blocks, skip connections.
    - **Difficulté** : ★★★☆☆ (Implémentation des connexions résiduelles).
    - https://arxiv.org/pdf/1506.02640 : Yolo
    - https://arxiv.org/pdf/1505.04597: Unet
    - https://arxiv.org/pdf/1502.03167 : BatchNorm

10. **LSTM** (1997) - *Hochreiter, Schmidhuber*  
    ["Long Short-Term Memory"](https://www.bioinf.jku.at/publications/older/2604.pdf)
    - **Concepts** : Portes (gates), mémoire à long terme.
    - **Difficulté** : ★★★★☆ (Dynamique des gradients récurrents).

11. **GANs** (2014) - *Goodfellow et al.*  
    ["Generative Adversarial Networks"](https://arxiv.org/abs/1406.2661)
    - **Concepts** : Jeu min-max, entraînement instable.
    - **Difficulté** : ★★★★☆ (Équilibre générateur/discriminateur).

---

### **Niveau 5 : Transformers et Attention (Expert)**
12. **Attention Seq2Seq** (2014) - *Bahdanau et al.*  
    ["Neural Machine Translation..."](https://arxiv.org/abs/1409.0473)
    - **Concepts** : Alignement soft, RNNs avec attention.
    - https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
    - **Difficulté** : ★★★★☆ (Mécanismes d'attention simples).

13. **Transformer** (2017) - *Vaswani et al.*  
    ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)

    - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
    -  https://jalammar.github.io/explaining-transformers/
    - 
    - **Concepts** : Attention multi-têtes, positional encoding.
    - **Difficulté** : ★★★★★ (Pas de convolutions ni récurrence).

14. **BERT** (2018) - *Devlin et al.*  
    ["BERT: Pre-training of Deep Bidirectional..."](https://arxiv.org/abs/1810.04805)

    - https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
    - **Concepts** : Masked LM, fine-tuning.
    - **Difficulté** : ★★★★★ (Pré-entraînement à grande échelle).

---

### **Niveau 6 : Théorie et Modèles de Pointe (Très Expert)**
15. **Diffusion Models** (2020) - *Ho et al.*  
    ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239)
    - **Concepts** : Processus Markovien, débruîtage itératif.
    - **Difficulté** : ★★★★★★ (Mathématiques avancées : SDEs, VLB).

16. **The Lottery Ticket Hypothesis** (2018) - *Frankle, Carbin*  
    ["The Lottery Ticket Hypothesis..."](https://arxiv.org/abs/1803.03635)
    - **Concepts** : Sous-réseaux efficaces, initialisation.
    - **Difficulté** : ★★★★★★ (Expériences de pruning complexes).

17. **AlphaGo** (2016) - *Silver et al.*  
    ["Mastering the Game of Go..."](https://www.nature.com/articles/nature16961)
    - **Concepts** : RL + MCTS, politiques de valeur.
    - **Difficulté** : ★★★★★★ (Intégration multi-domaines).

---

- [Distill pub](https://distill.pub/) (Explications visuelles des Transformers, GANs). 

---

Voici une liste d'**algorithmes fondamentaux** à implémenter pour maîtriser la programmation, la science des données, et l'informatique théorique. Pour chaque algorithme, je mentionne un papier ou une référence clé pour comprendre son fonctionnement et son contexte.

---

### **Algorithmes Classiques (Informatique/Programmation)**
1. **Tri Rapide (Quicksort)**
    - Papier original : *"Quicksort"* (1962) par Tony Hoare.
    - Pourquoi : Base de l'algorithmique récursive et de la complexité \(O(n \log n)\).

2. **Algorithme de Dijkstra (Plus court chemin)**
    - Référence : *"A Note on Two Problems in Connexion with Graphs"* (1959) par Edsger W. Dijkstra.
    - Application : Réseaux, navigation, graphes pondérés.

3. **FFT (Transformée de Fourier Rapide)**
    - Papier : *"An Algorithm for the Machine Calculation of Complex Fourier Series"* (1965) par J. W. Cooley et J. Tukey.
    - Pourquoi : Central en traitement du signal et imagerie.

4. **Algorithmes de Parcours (BFS/DFS)**
    - Référence : *"Introduction to Algorithms"* (Cormen et al.) pour la théorie.
    - Applications : IA (recherche dans des arbres), réseaux sociaux.

5. **Algorithme de Knuth-Morris-Pratt (KMP)**
    - Papier : *"Fast Pattern Matching in Strings"* (1977) par D. Knuth et al.
    - Pourquoi : Optimisation de la recherche de sous-chaînes.

---

### **Machine Learning/Data Science**
6. **Algorithme des k-moyennes (k-means)**
    - Papier original : *"Some methods for classification and analysis of multivariate observations"* (1967) par J. MacQueen.
    - Pourquoi : Clustering non supervisé.

7. **Régression Linéaire (Moindres Carrés)**
    - Référence : *"Gauss et la méthode des moindres carrés"* (historique), ou *"The Elements of Statistical Learning"* (Hastie et al.).
    - Base de l'apprentissage supervisé.

8. **Machine à Vecteurs de Support (SVM)**
    - Papier clé : *"Support-Vector Networks"* (1995) par Corinna Cortes et Vladimir Vapnik.
    - Pourquoi : Classification linéaire/non-linéaire via kernels.

9. **Réseaux de Neurones (Backpropagation)**
    - Papier fondateur : *"Learning representations by back-propagating errors"* (1986) par Rumelhart, Hinton, et Williams.
    - Base des réseaux profonds.

10. **Random Forest**
    - Référence : *"Random Forests"* (2001) par Leo Breiman.
    - Pourquoi : Méthode d'ensemble robuste pour la classification.

---

### **Optimisation/Recherche Opérationnelle**
11. **Simulated Annealing**
    - Papier : *"Optimization by Simulated Annealing"* (1983) par Kirkpatrick et al.
    - Application : Optimisation globale inspirée de la thermodynamique.

12. **Algorithme Génétique**
    - Référence : *"Genetic Algorithms in Search, Optimization, and Machine Learning"* (1989) par David E. Goldberg.
    - Pourquoi : Optimisation heuristique inspirée de l'évolution.

13. **Programmation Dynamique (ex: Problème du sac à dos)**
    - Référence : *"Dynamic Programming"* (1957) par Richard Bellman.
    - Applications : Optimisation de ressources, bioinformatique.

---

### **Cryptographie/Sécurité**
14. **RSA (Chiffrement asymétrique)**
    - Papier original : *"A Method for Obtaining Digital Signatures and Public-Key Cryptosystems"* (1978) par Rivest, Shamir, Adleman.
    - Pourquoi : Base de la cryptographie moderne.

15. **SHA-256 (Fonction de hachage)**
    - Référence : Norme NIST *"Secure Hash Standard"* (FIPS PUB 180-4).
    - Application : Blockchain, intégrité des données.

---

### **Traitement du Langage (NLP)**
16. **Algorithme de Viterbi**
    - Papier : *"Error bounds for convolutional codes and an asymptotically optimum decoding algorithm"* (1967) par Andrew Viterbi.
    - Pourquoi : Décodage de séquences (ex: étiquetage morpho-syntaxique).

17. **PageRank**
    - Papier original : *"The PageRank Citation Ranking: Bringing Order to the Web"* (1998) par Larry Page et Sergey Brin.
    - Application : Classement de pages web, analyse de graphes.

---

### **Algorithmes Modernes/IA**
18. **Q-Learning (Renforcement)**
    - Papier clé : *"Reinforcement Learning: A Survey"* (1996) par Leslie Kaelbling et al.
    - Pourquoi : Apprentissage par renforcement (ex: jeux, robots).

19. **Transformers (Attention Mechanism)**
    - Papier : *"Attention Is All You Need"* (2017) par Vaswani et al.
    - Pourquoi : Base des modèles de NLP comme GPT, BERT.

20. **Diffusion Models (ex: Stable Diffusion)**
    - Référence : *"Denoising Diffusion Probabilistic Models"* (2020) par Ho et al.
    - Application : Génération d'images, audio.

---
- https://github.com/sayakpaul/nanoDiT
-  https://papers.baulab.info
- https://proceedings.neurips.cc/paper_files/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf
- https://arxiv.org/pdf/2003.05991

- https://www.k-a.in/pyt-knn.html
- https://www.k-a.in/pyt-knn.html
-https://www.k-a.in/pyt-swiglu.html
- https://www.k-a.in/pyt-glu.html
- https://www.k-a.in/pyt-miniGPT.html
- https://www.k-a.in/pyt-comptr.html
- https://www.k-a.in/pyt-retro.html
- https://www.k-a.in/pyt-alibi.html
- https://www.k-a.in/pyt-rope.html
- https://www.k-a.in/pyt-transformerXL.html
- https://www.k-a.in/pyt-mha.html
- https://www.k-a.in/muon2.html
- https://www.k-a.in/muon.html
- https://www.k-a.in/muon1.html
- https://www.k-a.in/grpo-1B.html
- https://www.k-a.in/grpo.html

- https://github.com/SakanaAI/continuous-thought-machines/tree/main/examples
- chemistry: https://www.owlposting.com/p/generative-ml-in-chemistry-is-bottlenecked?open=false#§how-the-synthesis-bottleneck-manifests-in-ml
