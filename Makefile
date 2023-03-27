SHELL:=/bin/bash
splits := $(wildcard split/*)
splitparses := $(wildcard split-parse/*)

ID2X := data/id2mean-node-types-per-sentence data/id2mean-degree data/id2lexical-density data/id2mean-sentence-mean-lexical-diversity

data/descriptive-statistics.twitter: data/descriptive-statistics
data/descriptive-statistics.answer: data/descriptive-statistics
data/descriptive-statistics.speech: data/descriptive-statistics

data/descriptive-statistics: ./src/descriptive-statistics data/id2forum-and-proxies
	$^ $@

data/id2proxies: ./src/join  $(ID2X)
	$^ $@

data/id2forum-and-proxies: ./src/join data/sean-sherlock-unique-ids.tsv $(ID2X)
	./src/join <(./src/cols data/sean-sherlock-unique-ids.tsv id forum) $(ID2X) $@

data/id2forum: ./src/cols data/sean-sherlock-unique-ides.tsv

fig/mean-sentence-meaan-lexical-diversity-displot.png: ./src/displot data/id2mean-sentence-mean-lexical-diversity
	$^ mean-sentence-mean-lexical-diversity $@

fig/lexical-density-displot.png: ./src/displot data/id2lexical-density
	$^ lexical-density $@

fig/mean-node-types-per-sentence-displot.png: ./src/displot data/id2mean-node-types-per-sentence
	$^ mean-node-types-per-sentence $@

fig/mean-mean-degree-displot.png: ./src/displot data/id2mean-degree
	$^ mean-mean-degree $@

fig/mean-flat-degree-displot.png: ./src/displot data/id2mean-degree
	$^ mean-flat-degree $@

data/id2mean-degree: ./src/mean-degree data/id2node-degree-list-per-sentence
	$^ $@

data/id2node-degree-list-per-sentence: ./src/node-degree-list data/parses.pickle
	$^ $@

data/id2mean-node-types-per-sentence: ./src/id2mean-node-types-per-sentence data/id2node-type-list
	$^ $@

data/id2sentences: ./src/id2sentences data/parses.pickle
	$^ $@

data/id2node-type-list: ./src/node-type-list data/parses.pickle
	$^ $@

data/id2mean-sentence-mean-lexical-diversity: ./src/mean-sentence-mean-lexical-diversity data/stanza-lemmas-token2lex-score data/parses.pickle
	$^ $@

data/id2lexical-density: ./src/lexical-density data/stanza-lemmas-token2lex-score data/parses.pickle
	$^ $@

data/id2sentence-mean-lexical-diversity: ./src/sentence-mean-lexical-diversity data/stanza-lemmas-token2lex-score data/parses.pickle 
	$^ $@

data/word2lex-score: ./src/token2lexicality-score data/id2words-json
	$< words data/id2words-json $@

data/stanza-lemmas-token2lex-score: ./src/token2lexicality-score data/id2stanza-lemmas-json
	$< stanza-lemmas data/id2stanza-lemmas-json $@

data/parses.pickle: ./src/reduce-pickle parses.pickle data/sean-sherlock-unique-ids.tsv
	$^ $@

data/sean-sherlock-unique-ids.tsv: ./src/unique-texts seansherlock.csv
	$^ $@

data/id2mdwl.tsv: ./src/id2mean-document-word-length data/id2words-json
	$^ $@

data/id2words-json: ./src/pickle-to-id2words data/parses.pickle
	$^ $@

data/parses-pickle: ./src/reduce-pickle parses.pickle data/sean-sherlock-unique-ids.tsv
	$^ $@

data/id2stanza-lemmas-json: ./src/pickle-to-id2lemmas data/parses.pickle
	$^ $@

id2mean-tfidf-stanza-lemmas: ./src/id2mean-tfidf id2lemmas 
	$^ > $@
	
parses.pickle:
	./src/concat-pickled-dfs $@ split-parse/*

split-toks: $(foreach s,$(splitparses),split-toks/$(notdir $(s).toks))

split-toks/%.toks: split-toks/
	./src/pickle2toks split-parse/$* > $@

split-toks/:
	mkdir -p $@
split-parses: $(foreach s,$(splits),split-parse/$(notdir $(s).parse.pickle))
split-parse/%.parse.pickle: split/% split-parse/ 
	./src/parse-texts <(cat head $<) $@


split-parse/:
	mkdir -p $@

mk-splits: $(splits)
split/: no-head-seansherlock.csv
	mkdir -p split/
	split -l$$((`wc -l < $<`/20)) $< split/
	
id2parse.csv: id-seansherlock.csv
	./src/parse-texts $< 2> /dev/null | head

no-head-seansherlock.csv: id-seansherlock.csv
	tail -n +2 $< > $@

id-seansherlock.csv: seansherlock.csv
	./src/ids $< > $@

seansherlock.csv:
	cp ../comp-ling/seansherlock-final.csv $@

head: id-seansherlock.csv
	head -1 $< > $@
