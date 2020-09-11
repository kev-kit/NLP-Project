'''
	PROGETTO DI ESAME DI LINGUISTICA COMPUTAZIONALE DI KEVIN VANNI
****************************Programma 1**********************************
'''


from texttable import Texttable # lo utilizzo per stampare le tabelle (pip install texttable, documentazione: https://github.com/foutaise/texttable)

import sys 
import codecs
import nltk
import re

################### funzioni richieste ###################

# lunghezza media di: A) token in termini di caratteri; B) frasi in termini di token
def medie(corp_frasi, corp_token):
	mediaTok = round(float(len(corp_token))/len(corp_frasi), 2) 
	# la somma del numero di token per frase corrisponde al numero totale di token nel corpus cioe' len(corpus)
	
	acc = 0	
	for tok in corp_token:
		acc += len(tok)
	mediaChar = round(float(acc)/len(corp_token), 2) 

	return mediaTok, mediaChar

# restituisco la lista contenente il numero di hapax per porzioni incrementali di 1000 token
def hapax_1000(corpus, vocabolario):
	hapax = []
	acc = 0
	for i in range(len(corpus)):
		if i % 1000 == 0 and i != 0: 
			hapax.append(acc)
		if vocabolario[corpus[i]] == 1:
			acc += 1
	return hapax

# ricchezza lessicale attraverso TTR (Type/Token Ratio) sui primi 5000 token del corpus 
def TTR(corpus):
	cut_corpus = corpus[:]
	del(cut_corpus[4999:len(cut_corpus)])
	parole_tipo = sorted(cut_corpus)
	parole_tipo = set(cut_corpus) 		# parole tipo nei primi 5000 token  
	
	# parole unita' corrispondo a len(cut_corpus) cioe' 5000. 
	return round(float(len(parole_tipo))/len(cut_corpus), 2)

# calcolo frequenza relativa dell'elemento 
def freq_rel(elem, lenght):
	return round((float(elem)/lenght)*100, 2)

# calcolo numero medio dell'elemento per frase
def media_per_frase(elem, frasi): 
	return round(float(elem)/len(frasi), 2)


################### funzioni ausiliarie ###################

# leggo il file parametro e lo rendo accessibile all'analisi con python
def openFile(file): 
	fileInput = codecs.open(file, 'r', 'utf-8')
	return fileInput.read()

# tokenizzo il corpus raw in unita' minime pari alle frasi (sempre da decodificare)
def frasi_corpus(raw):
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	frasi = tokenizer.tokenize(raw)
	return frasi

# tokenizzo il corpus diviso in frasi in unita' minime
def token_corpus(corpus):
	tokens = []
	for frase in corpus:
		frase.encode('utf-8')
		tokens += nltk.word_tokenize(frase) 
	return tokens

# vocabolario del corpus con ogni voce accompagnata dalla sua frequenza
def vocabolario_corpus(corpus):
	ordinato = sorted(corpus)
	univoco = set(corpus)
	univoco = sorted(univoco)

	j = i = 0
	vocabolario = {} 		# creao il vocabolario come oggetto dizionario e ad ogni elemento assegno la sua frequenza

	while i < len(univoco):
		num_occ = 0
		while  (ordinato[j] == univoco[i]) and (j < len(ordinato)-1):
			num_occ += 1
			j += 1
		if (j == len(corpus)-1):
			vocabolario[univoco[i]] = num_occ+1
			# devo utilizzare questa dicitura in quanto controllare l'ultimo token darebbe indice j == len(obj) causando il controllo in una posizione non esistente e percio' errore
		else:
			vocabolario[univoco[i]] = num_occ
		i += 1

	return vocabolario

# corpus taggato tramite POS
def POS(corpus):
	taggato = [] 	# testo taggato applicando POS su di ogni frase (singolarmente)
	for frase in corpus:
		tmp = frase.encode('utf-8')
		tmp = nltk.word_tokenize(tmp)
		tmp = nltk.pos_tag(tmp)
		taggato += tmp
	return taggato

# frequenza assoluta di Sostantivi (NN, NNS, NNP, NNPS), Aggettivi (JJ, JJR, JJS), Verbi (VB, VBD, VBG, VBN, VBP, VBZ), Pronomi (PRP, PRP$, WP, WP$) in corpus
def freq_assoluta(corpus):
	sostantivi = aggettivi = verbi = pronomi = 0	
	for i in range(len(corpus)):
		if re.match(r'NN.*', corpus[i][1]):
			sostantivi += 1
		if re.match(r'JJ.*', corpus[i][1]):
			aggettivi += 1
		if re.match(r'VB.*', corpus[i][1]):
			verbi += 1
		if re.match(r'PRP.*|WP.*', corpus[i][1]):
			pronomi += 1
	return sostantivi, aggettivi, verbi, pronomi

# token comuni ad entrambi i corpus
def common_token(vocabolario1, vocabolario2):
	common = []
	for i in vocabolario1:
		if i in vocabolario2:
			common.append(i)
	return common

# funzione di confronto dei valori di lunghezza media di token per frase e caratteri per token
def confronto_media(nome_A, media_A, nome_B, media_B, argomento):
	if media_A > media_B:
		layout_punto2(nome_A, media_A, nome_B, media_B, argomento)
	elif media_A < media_B:
		layout_punto2(nome_B, media_B, nome_A, media_A, argomento)
	else:
		print "I corpus", nome_A, "e", nome_B, "hanno stessa lunghezza media di", argomento + ":", media_A

# funzione di confronto della lunghezza dei vocaboli
def confronto_voc(nome_A, voc_A, nome_B, voc_B, argomento):
	if voc_A > voc_B:
		layout_punto3_A(nome_A, voc_A, nome_B, voc_B, argomento)
	elif voc_A < voc_B:
		layout_punto3_A(nome_B, voc_B, nome_A, voc_A, argomento)
	else:
		print "I corpus", nome_A, "e", nome_B, "hanno stessa", argomento + ":", voc_A

################### funzioni di stampa ###################

# layout di stampa degli hapax
def print_hapax(hapax):
	j = 1
	for i in hapax:
		print "\t-", str(1000*j) + ":", i
		j += 1

# layout di stampa del punto 2
def layout_punto2(nome_mag, val_mag, nome_min, val_min, argomento):
	print "La lunghezza media di", argomento, "in", nome_mag, "e' maggiore rispetto a", nome_min + ":", val_mag, "vs", val_min

# layout di stampa della prima parte del punto 3 (vocabolario)
def layout_punto3_A(nome_mag, val_mag, nome_min, val_min, argomento):
	print "La", argomento, "di", nome_mag, "e' maggiore rispetto a", nome_min + ":", val_mag, "vs", val_min 

# layout di stampa della prima parte del punto 3 (hapax), utilizzo modulo texttable (che crea tabelle con caratteri ASCII) e str.format() (documentazione: https://docs.python.org/3/library/stdtypes.html#str.format e https://docs.python.org/3/library/string.html#formatspec)
def layout_punto3_B(nome_A, hapax_A, nome_B, hapax_B):
	table = Texttable()						# creo oggetto tabella
	table.__init__(0)						# imposto la max_width infinita
	raw_head = ["", nome_A, nome_B, ""]				# lista contenente le stringhe da inserire nel titolo della tabella non ancora formattati
	head_format = ["{:^6}", "{:^22}", "{:^22}", "{:^6}"]		# tipo di formattazione da applicare alle stringhe di raw_head
	head = []							# lista in cui inserire le stringhe di raw_head formattate
	for i, j in zip(raw_head, head_format):
		head.append(j.format(i))				# applico la formattazione
	table.header(head)						# definisco le stringhe titoli della tabella (passate tramite lista) 

	raw_subHead = ["token", "hapax", "incr", "hapax", "incr", "diff"]				# stringhe sottotitolo non formattate
	subHead_format = ["{:^6}", "{:^11}", "{:^11}", "{:^11}", "{:^11}", "{:^6}"]			# formattazione sottotitolo
	subHead = []											# stringhe sottotitolo formattate
	for i, j in zip(raw_subHead, subHead_format):
		subHead.append(j.format(i))
	subHead[1] += subHead[2]					# unisco le stringhe [1] e [2] le quali comporranno le il sottotitolo (devo fare questo per via della limitazione di texttable di inserire non piu' elementi della prima riga nelle righe successive)
	subHead[3] += subHead[4]					# come sopra per stringhe [3] e [4]
	del(subHead[2])							# elimino [2] che ho gia' unito alla stringa [1]
	del(subHead[3])							# elimino [3] che ho gia' unito alla stringa [2] ([3] e [2] sono rispettivamente [3] e [4] prima della eliminazione della stringa [2])
	table.add_row(subHead)						# aggiungo la riga sottotitolo alla tabella

	len_row = len(hapax_A) if len(hapax_A) > len(hapax_B) else len(hapax_B)				# numero righe contenenti dati
	for i in range(len_row):									# creo le righe in cui inserisco i dati
		# dati da inserire nella riga senza formattazione
		raw_row = [str(1000 * (i+1)),																																	# numero di token in esame 												
					"" if i > (len(hapax_A)-1) else str(hapax_A[i]), 																									# numero di hapax per token corpus A (la clausola if evita "out of range")
					"" if i > (len(hapax_A)-1) else ("+" + str((hapax_A[i] - hapax_A[i-1]) if i>0 else hapax_A[i])), 													# incremento rispetto a porzione di token precedente
					"" if i > (len(hapax_B)-1) else str(hapax_B[i]), 																									# numero di hapax per token corpus B
					"" if i > (len(hapax_B)-1) else ("+" + str((hapax_B[i] - hapax_B[i-1]) if i>0 else hapax_B[i])), 													# incremento rispetto a porzione di token precedente
					"" if i > (len(hapax_A)-1) or i > (len(hapax_B)-1) else (str(hapax_A[i] - hapax_B[i] if hapax_A[i] > hapax_B[i] else hapax_B[i] - hapax_A[i]))]		# differenza di hapax tra corpus A e B ogni porzione di token

		row_format = ["{:^6}", "{:^11}", "{:^11}", "{:^11}", "{:^11}", "{:^6}"]			# formattazione riga
		row = []																		# array che conterra riga di dati formattata
		for i, j in zip(raw_row, row_format):
			row.append(j.format(i))
		row[1] += row[2]							# stessi passaggi come riga subHead
		row[3] += row[4]							# //
		del(row[2])								# //	
		del(row[3])								# //
		table.add_row(row)							# aggiunta riga dati alla tabella

	print table.draw()								# stampa tabella

# layout di stampa della prima parte del punto 5 e 6
def layout_punto5e6(nome_A, cat_A, nome_B, cat_B):
	table = Texttable()								# creo oggetto tabella
	table.__init__(0)								# imposto la max_width infinita
	raw_head = ["", nome_A, nome_B]							# lista contenente le stringhe da inserire nel titolo della tabella non ancora formattati
	head_format = ["{:^12}", "{:^24}", "{:^24}"]					# tipo di formattazione da applicare alle stringhe di raw_head
	head = []									# lista in cui inserire le stringhe di raw_head formattate
	for i, j in zip(raw_head, head_format):
		head.append(j.format(i))						# applico la formattazione
	table.header(head)								# definisco le stringhe titoli della tabella (passate tramite lista) 

	raw_subHead = ["", "freq_rel %", "media", "freq_rel %", "media"]					# stringhe sottotitolo non formattate
	subHead_format = ["{:^12}", "{:^12}", "{:^12}", "{:^12}", "{:^12}"]					# formattazione sottotitolo
	subHead = []												# stringhe sottotitolo formattate
	for i, j in zip(raw_subHead, subHead_format):
		subHead.append(j.format(i))
	subHead[1] += subHead[2]					# unisco le stringhe [1] e [2] le quali comporranno le il sottotitolo (devo fare questo per via della limitazione di texttable di inserire non piu' elementi della prima riga nelle righe successive)
	subHead[3] += subHead[4]					# come sopra per stringhe [3] e [4]
	del(subHead[2])							# elimino [2] che ho gia' unito alla stringa [1]
	del(subHead[3])							# elimino [3] che ho gia' unito alla stringa [2] ([3] e [2] sono rispettivamente [3] e [4] prima della eliminazione della stringa [2])
	table.add_row(subHead)						# aggiungo la riga sottotitolo alla tabella

	catGram = ["sostantivi", "aggettivi", "verbi", "pronomi"] 
	j = 0
	for i in catGram:												# creo le righe in cui inserisco i dati
		# dati da inserire nella riga senza formattazione
		raw_row = [i, str(cat_A[j][0]), str(cat_A[j][1]), str(cat_B[j][0]), str(cat_B[j][1])]
		row_format = ["{:^12}", "{:^12}", "{:^12}", "{:^12}", "{:^12}"]						# formattazione riga
		row = []												# array che conterra riga di dati formattata
		for i, k in zip(raw_row, row_format):
			row.append(k.format(i))
		row[1] += row[2]								# stessi passaggi come riga subHead
		row[3] += row[4]								# //
		del(row[2])									# //	
		del(row[3])									# //
		table.add_row(row)								# aggiunta riga dati alla tabella
		j+=1

	print table.draw()									# stampa tabella


#						--- funzione main e rispettiva chiamata ---
def main(file1, file2):

	################### dichiarazione variabili ###################
	
	nomeCorpus_A = None				# nome del file1 utilizzato come corpus A
	raw_A = None					# corpus A non ancora soggetto ad analisi
	frasi_A = None					# corpus A diviso in frasi (da decodificare)
	
	# salvo i corpus divisi in frasi perche' e' richiesto il numero medio, per frasi, di sostantivi, aggettivi, ecc.

	token_A = None					# corpus A diviso in token (corrisponde a corpus A)
	lenght_A  = None 				# cardinalita' del corpus A |A|
	mediaTok_A = None				# media dei token per frase in corpus A
	mediaChar_A = None				# media caratteri per token del corpus A
	vocabolario_A = {}				# vocabolario del corpus A
	hapax_A = []					# hapax del corpus A per porzioni incrementali di 1000 token
	ttr_A = None					# valore type/token ratio del corpus A
	pos_A = None					# Part Of Speech Tagging su corpus A 
	sost_A = agg_A = ver_A = pron_A = None
	# frequenze assolute di sostantivi, aggettivi, verbi e pronomi del corpus A (variabili volutamente distinte da corpus B)

	nomeCorpus_B = None				# nome del file2 utilizzato come corpus B
	raw_B = None					# corpus B non ancora soggetto ad analisi
	frasi_B	= None					# corpus B diviso in frasi (da decodificare)
	token_B = None					# corpus B diviso in token (corrisponde a corpus B)
	lenght_B = None					# cardinalita' del corpus B |B|
	mediaTok_B = None				# media dei token per frase in corpus B
	mediaChar_B = None				# media caratteri per token del corpus B
	vocabolario_B = {}				# vocabolario del corpus B
	hapax_B = []					# hapax del corpus B per porzioni incrementali di 1000 token
	ttr_B = None					# valore type/token ratio del corpus B
	pos_B = None					# Part Of Speech Tagging su corpus B
	sost_B = agg_B = ver_B = pron_B = None
	# frequenze assolute di sostantivi, aggettivi, verbi e pronomi del corpus B (variabili volutamente distinte da corpus A)

	token_comuni = None				# parole tipo comuni ai due corpus 

	################### chiamata funzioni e inizializzazione variabili ###################

	nomeCorpus_A = file1
	raw_A = openFile(file1)				# funzioni ausiliarie

	nomeCorpus_B = file2
	raw_B = openFile(file2)				# funzioni ausiliarie

	frasi_A = frasi_corpus(raw_A) 			# corpus A diviso in frasi, funzioni ausiliarie
	frasi_B = frasi_corpus(raw_B)			# corpus B diviso in frasi, funzioni ausiliarie
	
	# inizio analisi
	
	token_A = token_corpus(frasi_A) 		# corpus A tokenizzato, funzioni ausiliarie
	token_B = token_corpus(frasi_B) 		# corpus B tokenizzato, funzioni ausiliarie
	
	lenght_A = len(token_A)
	lenght_B = len(token_B) 
	
	# calcolo medie
	
	mediaTok_A, mediaChar_A = medie(frasi_A, token_A) 		# funzioni richieste
	mediaTok_B, mediaChar_B = medie(frasi_B, token_B) 		# funzioni richieste

	# calcolo vocabolario e hapax

	vocabolario_A = vocabolario_corpus(token_A)			# funzioni ausiliarie
	hapax_A = hapax_1000(token_A, vocabolario_A) 			# funzioni richieste
	
	vocabolario_B = vocabolario_corpus(token_B)			# funzioni ausiliarie
	hapax_B = hapax_1000(token_B, vocabolario_B) 			# funzioni richieste

	token_comuni = common_token(vocabolario_A, vocabolario_B)	# funzioni ausiliarie

	# calcolo TTR
	ttr_A = TTR(token_A)		# funzioni richieste
	ttr_B = TTR(token_B)		# funzioni richieste

	# POS
	pos_A = POS(frasi_A) 		# funzioni ausiliarie
	pos_B = POS(frasi_B) 		# funzioni ausiliarie

	# frequenza assoluta sostantivi, aggettivi, verbi e pronomi
	sost_A, agg_A, ver_A, pron_A = freq_assoluta(pos_A)		# funzioni ausiliarie
	sost_B, agg_B, ver_B, pron_B = freq_assoluta(pos_B)		# funzioni ausiliarie
	'''
	attraverso frequenza assoluta posso calcolare: 
		- frequenze relative: |cat.gram|/len(corpus) (*100 ottengo il valore percentuale);
		- media cat.gram per frase: |cat.gram|/len(frasi_)
	'''	

	################### print valori e funzioni di confronto ###################

	print "PROGETTO DI ESAME DI LINGUISTICA COMPUTAZIONALE DI KEVIN VANNI\n****************************Programma 1**********************************"

	# stampo il pto 1
	print "\n--------------------------- PUNTO 1 ---------------------------\n"

	print "Il", nomeCorpus_A, "ha", len(frasi_A), "frasi su una lunghezza totale di", lenght_A, "token."
	print "Il", nomeCorpus_B, "ha", len(frasi_B), "frasi su una lunghezza totale di", lenght_B, "token."
	print "I due corpus hanno", len(token_comuni), "token in comune."

	# stampo il pto 2
	print "\n--------------------------- PUNTO 2 ---------------------------\n"

	# funzioni ausiliarie
	confronto_media(nomeCorpus_A, mediaTok_A, nomeCorpus_B, mediaTok_B, "token per frase")	
	confronto_media(nomeCorpus_A, mediaChar_A, nomeCorpus_B, mediaChar_B, "caratteri per token")

	# stampo il pto 3
	print "\n--------------------------- PUNTO 3 ---------------------------\n"

	confronto_voc(nomeCorpus_A, len(vocabolario_A), nomeCorpus_B, len(vocabolario_B), "lunghezza del vocabolario")		# funzioni ausiliarie

	print "\nIl loro numero di hapax per porzioni incrementali di 1000 hapax:\n"
	layout_punto3_B(nomeCorpus_A, hapax_A, nomeCorpus_B, hapax_B)		# funzioni di stampa

	# stampo il pto 4
	print "\n--------------------------- PUNTO 4 ---------------------------\n"
	
	print "TOKEN/TYPE Ratio:" 
	print "- La ricchezza lessicale del", nomeCorpus_A, "e':", ttr_A
	print "- La ricchezza lessicale del", nomeCorpus_B, "e':", ttr_B

	# stampo il pto 5
	print "\n--------------------------- PUNTI 5 e 6 ---------------------------\n"

	catGramm_A = [(freq_rel(sost_A, lenght_A), media_per_frase(sost_A, frasi_A)), (freq_rel(agg_A, lenght_A), media_per_frase(agg_A, frasi_A)), (freq_rel(ver_A, lenght_A), media_per_frase(ver_A, frasi_A)), (freq_rel(pron_A, lenght_A), media_per_frase(pron_A, frasi_A))]
	catGramm_B = [(freq_rel(sost_B, lenght_B), media_per_frase(sost_B, frasi_B)), (freq_rel(agg_B, lenght_B), media_per_frase(agg_B, frasi_B)), (freq_rel(ver_B, lenght_B), media_per_frase(ver_B, frasi_B)), (freq_rel(pron_B, lenght_B), media_per_frase(pron_A, frasi_A))]
	#catGramm_ lista in cui accoppio requqnza realtiva e media delle categorie grammaticali richieste
	
	layout_punto5e6(nomeCorpus_A, catGramm_A, nomeCorpus_B, catGramm_B)		# funzioni di stampa

	print "\n--------------------------- FINE Programma 1 ---------------------------\n"

main(sys.argv[1], sys.argv[2])
