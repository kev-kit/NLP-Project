

								Progetto di esame per il corso di Linguistica Computazionale


**Obiettivo**: Realizzazione di due programmi scritti in Python che utilizzino i moduli presenti in Natural Language Toolkit per leggere due file di testo in inglese, 
annotarli linguisticamente, confrontarli sulla base degli indici statistici richiesti ed estrarne le informazioni richieste.


Fasi realizzative: Creazione di due corpora in inglese, di almeno 5000 token ciascuno, contenenti testi estratti rispettivamente da blog di racconti di viaggio scritti da uomini e donne. I corpora sono salvati in due file di testo semplice in codifica utf-8.
Sviluppo di due programmi che prendono in input i due file da riga di comando, che li analizzano linguisticamente fino al Part-of-Speech tagging e che eseguono le seguenti operazioni.

**Programma 1** - Confrontare i due testi sulla base delle informazioni statistiche: 

	- numero di frasi e di token;
	- lunghezza media delle frasi in termini di token e la lunghezza media delle parole in termini di caratteri; 
	- grandezza del vocabolario e il numero di hapax all'aumentare del corpus per porzioni incrementali di 1000 token (1000 token, 2000 token, 3000 token, etc.); 
	- ricchezza lessicale calcolata attraverso la Type Token Ratio (TTR) sui primi 5000 token; 
	- distribuzione (in termini percentuali) di Sostantivi, Aggettivi, Verbi e Pronomi; 
	- numero medio di Sostantivi, Aggettivi, Verbi e Pronomi per frase.

**Programma 2** - Per ognuno dei due corpora estrarre le seguenti informazioni: 

	- estrarre e ordinare in ordine di frequenza decrescente, indicando anche la relativa frequenza: 
		◦ i 20 token più frequenti escludendo la punteggiatura; 
		◦ i 20 Aggettivi più frequenti; 
		◦ i 20 Verbi più frequenti; 
		◦ le 10 PoS (Part-of-Speech) più frequenti; 
		◦ i 10 trigrammi di PoS (Part-of-Speech) più frequenti;
	
	- estrarre e ordinare in ordine decrescente i 10 bigrammi di PoS (Part-of-Speech): 
		◦ con probabilità congiunta massima, indicando anche la relativa probabilità; 
		◦ con probabilità condizionata massima, indicando anche la relativa probabilità; 

	- creare un’unica lista con i 10 Sostantivi più frequenti contenuti nei blog maschili e i 10 Sostantivi più frequenti dei blog femminili e per ognuno di questi Sostantivi ordinare gli Aggettivi che li precedono rispetto alla forza associativa (calcolata in termini di Local Mutual Information);

	- dopo aver individuato e classificato le Entità Nominate (NE) presenti nel testo, estraerre: 
		◦ i 20 nomi propri di luogo più frequenti (tipi), ordinati per frequenza.
