'''
	PROGETTO DI ESAME DI LINGUISTICA COMPUTAZIONALE DI KEVIN VANNI
****************************Programma 2**********************************
'''

from texttable import Texttable # lo utilizzo per stampare le tabelle (pip install texttable, documentazione: https://github.com/foutaise/texttable)

import sys 
import codecs
import nltk
import re
import math

################### funzioni richieste ###################

# restituisce lista di lunghezza qnt contenente i qnt elementi piu' frequenti in parametro 1 (corpus)
def most_freq(corpus, qnt):
	distribuzione = nltk.FreqDist(corpus)	
	estratti_ord = distribuzione.most_common(qnt)				# most_common() restituisce lista di coppie (elem, freq)	
	return estratti_ord

# restituisce la massima probabilita' congiunta, freq_Bigramma/|C|, su lista di bigrammi
def max_probCongiunta(bigrammi):
	length =  len(bigrammi) + 1 						# (+1 vale come segnaposto)
	big_diversi = set(bigrammi)
	big_prob = []
	
	for i in big_diversi:
		tmp = float(bigrammi.count(i))
		big_prob.append((i, round(tmp/length, 3))) 			# questo calcola la max_prob_congiunta
	
	big_prob = sorted(big_prob, key = lambda elem: elem[1], reverse=True)	# dispongo lista in ordine decrescente
	del(big_prob[10:len(big_prob)])
	
	return big_prob

# restituisce la massima probabilita' condizionata, freqBigrammaA|B/freqA, su lista di bigrammi
def max_probCondizionata(bigrammi, pos):	
	big_diversi = set(bigrammi)
	big_prob = []
	
	for i in big_diversi:
		freqAB = float(bigrammi.count(i))
		freqA = float(pos.count(i[0])) 
		big_prob.append((i, round(freqAB/freqA, 3)))				# questo calcola la max_prob_condizionata
	
	big_prob = sorted(big_prob, key = lambda elem: elem[1], reverse = True)		# dispongo lista in ordine decrescente
	del(big_prob[10:len(big_prob)])
	
	return big_prob

# calcolo LMI tra sostantivo e gli aggettivi che lo precedono restituendo una lista di coppie (aggettivo, lmi)   
def LMI(sostantivo, lis_agg, corpusToken):
	lis_agg = list(set(lis_agg)) 											
	
	# in primo luogo realizzo lista "bigTok" con solo bigrammi di token
	bigrammi = list(nltk.bigrams(corpusToken))									# lista bigrammi di soli token
	agg_lmi = []													# lista di coppie (aggettivo, lmi)

	for i in lis_agg:
	# per ogni i, creo il bigramma (i, sostantivo) e ne conto l'occorrenza nella lista bigrammi
		bigCheck = creaBigramma(i, sostantivo) 									# restituisce bigramma da controllare; funzioni ausiliarie
		freqBig = bigrammi.count(bigCheck)
		numeratore = freqBig * (len(bigrammi) +1 )								# == len(corpusToken)
		denominatore = corpusToken.count(sostantivo) * corpusToken.count(i)
		MI = math.log((numeratore/denominatore), 2)								# calcolo MI
		lmi = round(freqBig * MI, 2)										# calcolo LMI
		agg_lmi.append((i, lmi))
	# caso lis_agg vuota
	if lis_agg == []:
		agg_lmi.append(("nessun aggettivo precede il sostantivo", ""))
	
	agg_lmi = sorted(agg_lmi, key = lambda elem: elem[1], reverse=True)
	return agg_lmi

# estraggo le 20 Entita' Nominate GPE piu' frequenti
def name_entity(pos):
	tree_net = nltk.ne_chunk(pos)
	
	gpe_entity = []									# lista in cui inserisco le entita' GPE
	for nodo in tree_net:
		NE = ''									# entita' nominata
		if hasattr(nodo, 'label'):
				if nodo.label() == "GPE":				# estraggo tutte le GPE
					for i in nodo.leaves():
						NE += i[0] + " " 			# devo aggiungere " " per dividere i nomi propi composti da piu' parole (cosi facendo ho per ogni gpe uno spazio in fondo alla parola) 
					gpe_entity.append(NE)				# in gpe_entity non mi interessa aggiungere la label
					
	gpe_freqMag = most_freq(gpe_entity, 20)	 	# estraggo i 20 nomi propi di luogo piu' frequenti
	return gpe_freqMag



################### funzioni ausiliarie ###################

def tokenizzatore(file):
	fileInput = codecs.open(file, 'r', 'utf-8')
	raw = fileInput.read()
	raw = re.sub(u'\ufeff', '', raw, 1)	 					# elimina il carattere byte order mark (BOM) se occorre all'inizio del documento
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	frasi = tokenizer.tokenize(raw)
	token = []
	for frase in frasi:
		frase.encode('utf-8')
		token += nltk.word_tokenize(frase)
	return token

# Tag corpus con POS
def POS(corpus):
	return nltk.pos_tag(corpus) 	# corpus intero taggato

# splitter restituisce lista contenente corpus scevro degli elementi che non soddisfano la re.match() 
def splitter(corpus, chiave): 
	lis_tmp = []
	for i in corpus:
		if re.match(chiave, i[1]):
			lis_tmp.append(i[0])
	return lis_tmp

# funzioni per calcolo LMI:
# A) creo lista con sostantivi in comune tra i due corpus
def crea_commonSost(pos_A, pos_B):
	sost = []
	only_sost = []

	sost = most_freq(splitter(pos_A, r'NN.*'), 10)			# utilizzo most_freq() sul corpus femminile taggato con pos e ottengo lista con coppia ((elem, pos_elem), freq) 
	for i in sost:
		only_sost.append(i[0])					# dalla lista ottenuta con most_freq() prendo solo l'elemento della coppia che corrisponde al sostantivo
	sost = most_freq(splitter(pos_B, r'NN.*'), 10)			# ripeto per il corpus maschile
	for i in sost:
		only_sost.append(i[0])

	return only_sost

# B) creo bigramma aggettivo - sostantivo su cui devo calcolare LMI  
def creaBigramma(agg, sost):
	bigramma = agg + " " + sost
	bigramma = nltk.word_tokenize(bigramma)
	bigramma = list(nltk.bigrams(bigramma))
	return bigramma[0]

# C) funzione in cui inizializzo gli elementi che mi servono per calcolare LMI e chiamo la funzione che effettivamente calcola la LMI
def lmi_aux(sostantivi, bigrammi_A, bigrammi_B, corpusToken_A, corpusToken_B):
	calcolo_lmi = []						# lista contenente triple (sostantivo, lista con aggettivi precedenti a sostantivo e relativa lmi per corpus A, lista con aggettivi precedenti a sostantivo e relativa lmi per corpus B)
	
	for i in sostantivi:
		corp_A = None						# lista contenente coppie (aggettivo, lmi) dell'aggettivo precedente al sostantivo e loro lmi per corpus A
		corp_B = None						# lista contenente coppie (aggettivo, lmi) dell'aggettivo precedente al sostantivo e loro lmi per corpus B
		agg = []						# lista di soli aggettivi

		for j in bigrammi_A:
			if j[1][0] == i and re.match(r'JJ.*', j[0][1]):
				agg.append(j[0][0])
		# calcolo LMI su i e suoi aggetivi per corpus femminile
		corp_A = LMI(i, agg, corpusToken_A) 			# restitusce una lista di coppie (agg, lmi)
		
		agg = []						# azzeramento lista aggettivi, agg = []

		for j in bigrammi_B:
			if j[1][0] == i and re.match(r'JJ.*', j[0][1]):
				agg.append(j[0][0])
		# calcolo LMI su i e suoi aggetivi per corpus maschile
		corp_B = LMI(i, agg, corpusToken_B) 

		calcolo_lmi.append((i, corp_A, corp_B))
	return calcolo_lmi

################### funzioni di stampa ###################

# layout di stampa della prima parte del punto 1
def layout_punto1(nome_A, elem_A, nome_B, elem_B, dim, argomento):
	table = Texttable()						# creo oggetto tabella
	table.__init__(0)						# imposto la max_width infinita
	raw_head = ["", nome_A, nome_B]					# lista contenente le stringhe da inserire nel titolo della tabella non ancora formattati
	head_format = ["{:^7}", "{:^24}", "{:^24}"]			# tipo di formattazione da applicare alle stringhe di raw_head
	head = []							# lista in cui inserire le stringhe di raw_head formattate
	for i, j in zip(raw_head, head_format):
		head.append(j.format(i))				# applico la formattazione
	table.header(head)						# definisco le stringhe titoli della tabella (passate tramite lista) 

	raw_subHead = ["rango", argomento, "freq", argomento, "freq"]				# stringhe sottotitolo non formattate
	subHead_format = ["{:^7}", "{:^12}", "{:^12}", "{:^12}", "{:^12}"]			# formattazione sottotitolo
	subHead = []										# stringhe sottotitolo formattate
	for i, j in zip(raw_subHead, subHead_format):
		subHead.append(j.format(i))
	subHead[1] += subHead[2]					# unisco le stringhe [1] e [2] le quali comporranno il sottotitolo (devo fare questo per via della limitazione di texttable di inserire non piu' elementi della prima riga nelle righe successive)
	subHead[3] += subHead[4]					# come sopra per stringhe [3] e [4]
	del(subHead[2])							# elimino [2] che ho gia' unito alla stringa [1]
	del(subHead[3])							# elimino [3] che ho gia' unito alla stringa [2] ([3] e [2] sono rispettivamente [3] e [4] prima della eliminazione della stringa [2])
	table.add_row(subHead)						# aggiungo la riga sottotitolo alla tabella
 
	j = 0
	for i in range(dim):												# creo le righe in cui inserisco i dati
		# dati da inserire nella riga senza formattazione
		raw_row = [i+1, str(elem_A[j][0]), str(elem_A[j][1]), str(elem_B[j][0]), str(elem_B[j][1])]
		row_format = ["{:^7}", "{:^12}", "{:^12}", "{:^12}", "{:^12}"]						# formattazione riga
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

# layout di stampa per punti in cui e' richiesto di stampare trigrammi e bigrammi
def layout_trig_e_big(nome_A, elem_A, nome_B, elem_B, dim, argomento):
	table = Texttable()								# creo oggetto tabella
	table.__init__(0)								# imposto la max_width infinita
	raw_head = ["", nome_A, nome_B]							# lista contenente le stringhe da inserire nel titolo della tabella non ancora formattati
	head_format = ["{:^7}", "{:^22}", "{:^22}"]					# tipo di formattazione da applicare alle stringhe di raw_head
	head = []									# lista in cui inserire le stringhe di raw_head formattate
	for i, j in zip(raw_head, head_format):
		head.append(j.format(i))						# applico la formattazione
	table.header(head)								# definisco le stringhe titoli della tabella (passate tramite lista) 

	raw_subHead = ["rango", argomento, "freq", argomento, "freq"]				# stringhe sottotitolo non formattate
	subHead_format = ["{:^7}", "{:^15}", "{:^15}", "{:^15}", "{:^15}"]			# formattazione sottotitolo
	subHead = []										# stringhe sottotitolo formattate
	for i, j in zip(raw_subHead, subHead_format):
		subHead.append(j.format(i))
	subHead[1] += subHead[2]					# unisco le stringhe [1] e [2] le quali comporranno le il sottotitolo (devo fare questo per via della limitazione di texttable di inserire non piu' elementi della prima riga nelle righe successive)
	subHead[3] += subHead[4]					# come sopra per stringhe [3] e [4]
	del(subHead[2])							# elimino [2] che ho gia' unito alla stringa [1]
	del(subHead[3])							# elimino [3] che ho gia' unito alla stringa [2] ([3] e [2] sono rispettivamente [3] e [4] prima della eliminazione della stringa [2])
	table.add_row(subHead)						# aggiungo la riga sottotitolo alla tabella
 
	j = 0
	for i in range(dim):								# creo le righe in cui inserisco i dati
		# dati da inserire nella riga senza formattazione
		tri_A = ""
		tri_B = ""
		for k in elem_A[j][0]: tri_A += k + " "					# stringa contenente i trigrammi del corpus A
		for k in elem_B[j][0]: tri_B += k + " "					# stringa contenente i trigrammi del corpus B

		raw_row = [i+1, tri_A, str(elem_A[j][1]), tri_B, str(elem_B[j][1])]
		row_format = ["{:^7}", "{:^15}", "{:^15}", "{:^15}", "{:^15}"]		# formattazione riga
		row = []								# array che conterra riga di dati formattata
		for i, k in zip(raw_row, row_format):
			row.append(k.format(i))
		row[1] += row[2]							# stessi passaggi come riga subHead
		row[3] += row[4]							# //
		del(row[2])								# //	
		del(row[3])								# //
		table.add_row(row)							# aggiunta riga dati alla tabella
		j+=1

	print table.draw()								# stampa tabella

# layout di stampa per punto 3
def layout_punto3(nome_A, nome_B, lmi):
	for i in lmi:
		print "SOSTANTIVO", '"' + i[0] + '"'
		print "\n\t- AGGETTIVI in", nome_A + ":"
		
		str_lmi = ''								# stringa contenenti aggettivi e lmi di bigramma aggettivo-sostantivo 
		for j in i[1]: str_lmi += j[0] + ", " + str(j[1]) + " | "

		print "\n\t\t", str_lmi, "\n"
		print "\n\t- AGGETTIVI in", nome_B + ":"
		
		str_lmi = ''
		for j in i[2]: str_lmi += j[0] + ", " + str(j[1]) + " | "

		print "\n\t\t", str_lmi, "\n"


# layout di stampa per punto 4 (funzione ad-hoc per lunghezza caratteri GPE)
def layout_punto4(nome_A, elem_A, nome_B, elem_B, dim):
	table = Texttable()							# creo oggetto tabella
	table.__init__(0)							# imposto la max_width infinita
	raw_head = ["rango", nome_A, nome_B]					# lista contenente le stringhe da inserire nel titolo della tabella non ancora formattati
	head_format = ["{:^7}", "{:^22}", "{:^22}"]				# tipo di formattazione da applicare alle stringhe di raw_head
	head = []								# lista in cui inserire le stringhe di raw_head formattate
	for i, j in zip(raw_head, head_format):
		head.append(j.format(i))					# applico la formattazione
	table.header(head)							# definisco le stringhe titoli della tabella (passate tramite lista) 
 
	j = 0
	for i in range(dim):													# creo le righe in cui inserisco i dati
		# dati da inserire nella riga senza formattazione
		raw_row = [i+1, str(elem_A[j][0]), str(elem_A[j][1]), str(elem_B[j][0]), str(elem_B[j][1])]
		row_format = ["{:^7}", "{:^16}", "{:^16}", "{:^16}", "{:^16}"]							# formattazione riga
		row = []													# array che conterra riga di dati formattata
		for i, k in zip(raw_row, row_format):
			row.append(k.format(i))
		row[1] += row[2]								# stessi passaggi come riga subHead
		row[3] += row[4]								# //
		del(row[2])									# //	
		del(row[3])									# //
		table.add_row(row)								# aggiunta riga dati alla tabella
		j+=1

	table.set_deco(Texttable.VLINES | Texttable.HEADER)				# imposto lo stile delle righe e colonne
	print table.draw()								# stampa tabella


#						--- funzione main e rispettiva chiamata ---
def main(file1, file2):

	################### dichiarazione variabili ###################

	nomeCorpus_A = None					# nome del file1 utilizzato come corpus A
	token_A = None						# corpus A diviso in token (corrisponde a corpus A)
	lenght_A  = None 					# cardinalita' del corpus A |A|
	pos_A = None						# Part Of Speech Tagging su corpus A 
	token20_A = agg20_A = ver20_A = []			# lista dei 20 token, aggettivi e verbi piu' frequenti nel corpus A
	listaPOS_A = []						# lista contenente solo POS del corpus A
	pos10_A = []						# lista dei 10 POS piu' frequenti nel corpus A
	trigrammiPos_A = []					# lista di trigrammi di POS del corpus A
	trigPos10_A = []					# lista dei 10 trigrammi di POS piu' frequenti nel corpus A	
	bigrammiPos_A = []					# lista di bigrammi di POS del corpus A
	bigCong10_A = bigCond10_A = []				# liste dei 10 bigrammi di POS del corpus A con probabilita' congiunta massima e probabilita' condizionata massima
	sost10_A = []						# lista dei 10 sostantivi piu' frequenti nel corpus A
	NE_gpe20_A = []						# lista delle 20 GPE piu' frequenti nel corpus A

	nomeCorpus_B = None					# nome del file2 utilizzato come corpus B
	token_B = None						# corpus B diviso in token (corrisponde a corpus B)
	lenght_B = None						# cardinalita' del corpus B |B|
	pos_B = None						# Part Of Speech Tagging su corpus B
	token20_B = agg20_B = ver20_B = []			# i 20 token, aggettivi e verbi piu' frequenti nel corpus B
	listaPOS_B = []						# lista contenente solo POS del corpus B
	pos10_B = []						# lista dei 10 POS piu' frequenti nel corpus B
	trigrammiPos_B = []					# lista di trigrammi di POS del corpus B
	trigPos10_B = []					# lista dei 10 trigrammi di POS piu' frequenti nel corpus B
	bigrammiPos_B = []					# lista di bigrammi di POS del corpus B
	bigCong10_B = bigCond10_B = []				# liste dei 10 bigrammi di POS del corpus B con probabilita' congiunta massima e probabilita' condizionata massima
	sost10_B = []						# lista dei 10 sostantivi piu' frequenti nel corpus B
	NE_gpe20_B = []						# lista delle 20 GPE piu' frequenti nel corpus B

	# variabili per calcolo LMI
	common_sost = []					# lista sostantivi comuni ai corpus
	bigrTOKENePOS_A = None  				# bigrammi (token - pos , token - pos) del corpus A	
	bigrTOKENePOS_B = None 					# bigrammi (token - pos , token - pos) del corpus B
	lmi_sost_agg = [] 					# lista di triple (sostantivo, [lista aggettivi corpus femminile che precedono sostantivo e relativa LMI], [lista aggettivi corpus maschile che precedono sostantivo e relativa LMI])

	################### chiamata funzioni e inizializzazione variabili ###################

	nomeCorpus_A = file1
	token_A = tokenizzatore(file1)
	lenght_A = len(token_A)
	
	nomeCorpus_B = file2
	token_B = tokenizzatore(file2)
	lenght_B = len(token_B)

	pos_A = POS(token_A)
	pos_B = POS(token_B)

	# estraggo i 20 token piu' frequenti escludendo la punteggiatura
	token20_A = most_freq(splitter(pos_A, r'\w'), 20)			# most_freq e' funzione richiesta 
	token20_B = most_freq(splitter(pos_B, r'\w'), 20)

	# estraggo i 20 aggettivi piu' frequenti r'^JJ.*' 
	agg20_A = most_freq(splitter(pos_A, r'JJ.*'), 20) 
	agg20_B = most_freq(splitter(pos_B, r'JJ.*'), 20)

	# estraggo i 20 verbi piu' frequenti r'^VB[DGNPZ]*'
	ver20_A = most_freq(splitter(pos_A, r'VB[DGNPZ]*'), 20)
	ver20_B = most_freq(splitter(pos_B, r'VB[DGNPZ]*'), 20)  

	# creo liste di soli elementi POS
	for i in pos_A: listaPOS_A.append(i[1])
	for i in pos_B: listaPOS_B.append(i[1])

	# estraggo i 10 POS piu' frequenti
	pos10_A = most_freq(listaPOS_A, 10)
	pos10_B = most_freq(listaPOS_B, 10) 
	
	# estraggo i 10 trigrammi di POS piu' frequenti su corpus A
	trigrammiPos_A = list(nltk.trigrams(listaPOS_A)) 
	trigPos10_A = most_freq(trigrammiPos_A, 10)

	# estraggo i 10 trigrammi di POS piu' frequenti su corpus B
	trigrammiPos_B = list(nltk.trigrams(listaPOS_B)) 
	trigPos10_B = most_freq(trigrammiPos_B, 10) 

	# calcolo max probabilita' congiunta e condizionata su i bigrammi di POS del corpus A
	bigrammiPos_A = list(nltk.bigrams(listaPOS_A))
	bigCong10_A = max_probCongiunta(bigrammiPos_A)									# funzioni richieste
	bigCond10_A = max_probCondizionata(bigrammiPos_A, listaPOS_A)							# funzioni richieste
	
	# calcolo max probabilita' congiunta e condizionata su i bigrammi di POS del corpus B
	bigrammiPos_B = list(nltk.bigrams(listaPOS_B))						
	bigCong10_B = max_probCongiunta(bigrammiPos_B)									# funzioni richieste
	bigCond10_B = max_probCondizionata(bigrammiPos_B, listaPOS_B)							# funzioni richieste

	# calcolo LMI su i 20 sostantivi piu' frequenti, dei corpus A e B, e i loro aggettivi precedenti  
		# operazioni ausiliarie
	common_sost = crea_commonSost(pos_A, pos_B)									# funzioni ausiliarie
	bigrTOKENePOS_A = list(nltk.bigrams(pos_A)) 
	bigrTOKENePOS_B = list(nltk.bigrams(pos_B))
		# LMI
	lmi_sost_agg = lmi_aux(common_sost, bigrTOKENePOS_A, bigrTOKENePOS_B, token_A, token_B)			# funzioni ausiliarie
	# alcuni sostantivi si possono ripetere perche' risultano i piu' utilizzati in entrambi i corpus 
	lmi_sost_agg = sorted(lmi_sost_agg, key = lambda elem: elem[0])						# ordina in modo alfabetico

	# estraggo le 20 Entita' Nominate GPE piu' frequenti del corpus A
	NE_gpe20_A = name_entity(pos_A)										# funzioni richieste

	# estraggo le 20 Entita' Nominate GPE piu' frequenti del corpus A
	NE_gpe20_B = name_entity(pos_B)										# funzioni richieste

	################### print valori e funzioni di confronto ###################

	print "PROGETTO DI ESAME DI LINGUISTICA COMPUTAZIONALE DI KEVIN VANNI\n****************************Programma 2**********************************"

	# stampo il pto 1
	print "\n--------------------------- PUNTO 1 ---------------------------\n"
	
	print " - 20 TOKEN piu' frequenti in corpus femminile e maschile (punteggiatura esclusa)\n"
	layout_punto1(nomeCorpus_A, token20_A, nomeCorpus_B, token20_B, 20, "token")				# funzioni di stampa
	
	print "\n - 20 AGGETTIVI piu' frequenti in corpus femminile e maschile\n"
	layout_punto1(nomeCorpus_A, agg20_A, nomeCorpus_B, agg20_B, 20, "aggettivi")				# funzioni di stampa

	print "\n - 20 VERBI piu' frequenti in corpus femminile e maschile\n"
	layout_punto1(nomeCorpus_A, ver20_A, nomeCorpus_B, ver20_B, 20, "verbi")				# funzioni di stampa

	print "\n - 10 POS piu' frequenti in corpus femminile e maschile\n"
	layout_punto1(nomeCorpus_A, pos10_A, nomeCorpus_B, pos10_B, 10, "POS")					# funzioni di stampa

	print "\n - 10 TRIGRAMMI di POS piu' frequenti in corpus femminile e maschile\n"
	layout_trig_e_big(nomeCorpus_A, trigPos10_A, nomeCorpus_B, trigPos10_B, 10, "trig_POS")			# funzioni di stampa (ad-hoc per trigrammi e bigrammi)

	# stampo il pto 2
	print "\n--------------------------- PUNTO 2 ---------------------------\n"

	print " - 10 BIGRAMMI di POS con PROBABILITA' CONGIUNTA MASSIMA in corpus femminile e maschile\n"
	layout_trig_e_big(nomeCorpus_A, bigCong10_A, nomeCorpus_B, bigCong10_B, 10, "big_POS")			# funzioni di stampa

	print "\n - 10 BIGRAMMI di POS con PROBABILITA' CONDIZIONATA MASSIMA in corpus femminile e maschile\n"
	layout_trig_e_big(nomeCorpus_A, bigCond10_A, nomeCorpus_B, bigCond10_B, 10, "big_POS")			# funzioni di stampa

	# stampo il pto 3
	print "\n--------------------------- PUNTO 3 ---------------------------\n"

	print " - Local Mutual Information calcolata su i 20 sostantivi con maggior frequenza dei due corpus e gli aggettivi ad essi precedenti\n"
 	layout_punto3(nomeCorpus_A, nomeCorpus_B, lmi_sost_agg)							# funzioni di stampa

	# stampo il pto 4
	print "--------------------------- PUNTO 4 ---------------------------\n"

	print " - 20 Entita' Nominate GPE con frequenza maggiore nei corpus femminile e maschile\n"
	layout_punto4(nomeCorpus_A, NE_gpe20_A, nomeCorpus_B, NE_gpe20_B, 20)					# funzioni di stampa

	print "\n\n--------------------------- FINE Programma 2 ---------------------------\n"


main(sys.argv[1], sys.argv[2])

