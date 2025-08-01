coNAtree = [ 
          "function", [
          "makesprotein", ["ribosomalrna", "trna"], 
          "nazyme", ["ribonucleasep", "group1intron", "group2intron", "glmsribozyme", "vsribozyme", "hairpin", "hammerhead", "hdvribozyme", "selfreplicating" ],
          "riboswitch", ["aaswitch", "abswitch", "ionswitch", "purineswitch", "tboxswitch", "vitaminswitch"],
          "aptamer", ["fluoroaptamer", "thrombinaptamer"],
          "translatemod", ["srprna", "tmrna", "rnaires", "frameshift"],
          "transcription", ["antiterminator", "natransfactor", ["hivtar","7sksnrna" ]],
           "posttranscription", ["snrna", "snorna", "cellsorting", "altsplice"], 
           "replication",
           "telomere", "virusna",
           "otherna" , [ "arrpofrna", "golldrna", "olerna", "raiarna", "roolrna" ] ],

          "nastructure",  
           [ "double", ["aform", "bform", "zform"], "parallel", ["imotif"],"triple", "quadruplex", "multiplex", "holliday",
          "feature", ["founding", "dodecamer", "cyclic", "bulge", "mispair", ["mispairaa", "mispairac", "mispairag",  "mispaircc", "mispairct", "mispairgg", "mispairgt", "mispairtt" ] ], 
           "designed", "ornate", ["ornatemulti"] ] ] ;



NAKBnadict = {
"function":      {"name":"function",                    
                    "hierarchy":"function", 
                    "descr":"functional role of nucleic acid" },
"makesprotein":    {"name":"protein synthesis", 
                    "hierarchy":"function > makesprotein", 
                    "descr":"component of protein synthesis translational machinery" },
"ribosomalrna":    {"name":"ribosomal RNA",        
                    "hierarchy":"function > makesprotein > ribosomalrna", 
                    "descr":"RNA component of ribosome, large or small subunit" },
"trna":            {"name":"transfer RNA",         
                    "hierarchy":"function > makesprotein > trna", 
                    "descr":"transfers an amino acid to growing protein chain" },
"codingrna":       {"name":"messenger RNA (mRNA)",         /* not shown in tree */
                    "hierarchy":"function > makesprotein > codingrna", 
                    "descr":"encodes gene to be translated into protein" },

"translatemod":    {"name":"translation regulating",   
                    "hierarchy":"function > translatemod", 
                    "descr":"mRNA element or RNA complex component that regulates protein synthesis" },
"tmrna":        {"name":"transfer-messenger (tmRNA)", 
                    "hierarchy":"function > translatemod > tmrna", 
                    "descr":"component of bacterial complex that recycles stalled ribosomes" },
"rnaires":         {"name":"IRES element",                
                    "hierarchy":"function > translatemod > rnaires", 
                    "descr":"<u>I</u>nternal <u>R</u>ibosome <u>E</u>ntry <u>S</u>ite, enables eukaryotic translation without mRNA 5&prime;cap" },
"frameshift":      {"name":"frameshifting element",       
                    "hierarchy":"function > translatemod > frameshift", 
                    "descr":"mRNA element that causes shift in codon triplet reading frame" },
"srprna":          {"name":"SRP RNA",                     
                    "hierarchy":"function > translatemod > srprna", 
                    "descr":"component of signal recognition particle that recognizes protein secretion signal" },

"replication": {"name":"replication regulating", 
                    "hierarchy":"function > replication", 
                    "descr":"nucleic acid element or complex component that regulates genomic or plasmid replication" },

"transcription": {"name":"transcription regulating", 
                    "hierarchy":"function > transcription", 
                    "descr":"mRNA element or RNA complex component that regulates transcription" },
"antiterminator": {"name":"antitermination element", 
                    "hierarchy":"function > transcription > antiterminator", 
                    "descr":"mRNA element that changes structure in response to binding by a transcription factor (e.g. TRAP)" },
"riboswitch":       {"name":"riboswitch",       
                    "hierarchy":"function > riboswitch", 
                    "descr":"natural or synthetic cis-acting gene expression-modulating mRNA element that responds to binding by a molecular effector" },
"aaswitch":       {"name":"amino-acid riboswitch",       
                    "hierarchy":"function > riboswitch > aaswitch", 
                    "descr":"changes structure in response to binding by an amino acid (e.g., glycine, glutamine)" },
"abswitch":       {"name":"antibiotic riboswitch",       
                    "hierarchy":"function > riboswitch > abswitch", 
                    "descr":"changes structure in response to binding by an antibiotic (e.g., neomycin)" },
"ionswitch":       {"name":"ion riboswitch",       
                    "hierarchy":"function > riboswitch > ionswitch", 
                    "descr":"changes structure in response to binding by an ion (e.g., Mg2+,F-)" },
"purineswitch":       {"name":"purine-compound riboswitch",       
                    "hierarchy":"function > riboswitch > purineswitch", 
                    "descr":"changes structure in response to binding by a purine-containing compound (e.g., guanine, hypoxanthine, cyclic-di-GMP)" },
"tboxswitch":       {"name":"tRNA riboswitch",       
                    "hierarchy":"function > riboswitch > tboxswitch", 
                    "descr":"changes structure in response to binding by tRNA" },
"vitaminswitch":    {"name":"enzyme cofactor riboswitch",       
                    "hierarchy":"function > riboswitch > vitaminswitch", 
                    "descr":"changes structure in response to binding by an enzyme co-factor (e.g., FMN, SAM, Thiamine, Folate)" },


"aptamer":          {"name":"aptamer",       
                    "hierarchy":"function > aptamer", 
                    "descr":"domain within a riboswitch--or synthetic RNA or DNA--that changes structure in response to binding by a molecular effector" },
"fluoroaptamer":    {"name":"fluorescing aptamer",       
                    "hierarchy":"function > aptamer > fluoroaptamer", 
                    "descr":"binds to fluorescent small molecule effector" },
"thrombinaptamer":    {"name":"thrombin responsive aptamer",       
                    "hierarchy":"function > aptamer > thrombinaptamer", 
                    "descr":"changes structure in response to binding by thrombin" },


"natransfactor": {"name":"trans-acting transcriptional regulator", 
                    "hierarchy":"function > transcription > natransfactor", 
                    "descr":"nucleic acid that acts in trans as a regulator of transcription" },
"hivtar": {"name":"activation response element",  
                    "hierarchy":"function > transcription > natransfactor > hivtar", 
                    "descr":"<u>T</u>rans-<u>A</u>ctivation <u>R</u>esponse element, stimulates transcription from viral promoter" },
"7sksnrna": {"name":"7SK RNA", 
                    "hierarchy":"function > transcription > natransfactor > 7sksnrna", 
                    "descr":"regulates transcription elongation via inhibition of cyclin-dependent kinase" },


"posttranscription": {"name":"post-transcriptional processing", 
                    "hierarchy":"function > posttranscription", 
                    "descr":"nucleic acid involved in processing RNA transcripts" },
"snrna": {"name":"small nuclear RNA (snRNA)", 
                    "hierarchy":"function > posttranscription > snrna", 
                    "descr":"RNA component of spliceosome or related mRNA-processing complex (e.g. U1-U7)" },
"snorna": {"name":"small nucleolar RNA (snoRNA)", 
                    "hierarchy":"function > posttranscription > snorna", 
                    "descr":"RNA guide for enzymatic modification of RNAs, e.g., methylation, pseudouridylation, cleavage" },
"cellsorting": {"name":"cellular localization element", 
                    "hierarchy":"function > posttranscription > cellsorting", 
                    "descr":"mRNA element that encodes active transport to a specific cellular region" },
"altsplice": {"name":"alternative splicing element", 
                    "hierarchy":"function > posttranscription > altsplice", 
                    "descr":"mRNA element that controls alternative splicing (e.g. tau exon)" },


"nazyme": {"name": "catalytic",
                    "hierarchy":"function > nazyme", 
                    "descr":"nucleic acid enzyme, ribozyme or dnazyme" },

"group1intron": {"name":"group I intron ribozyme",  
                    "hierarchy":"function > nazyme > group1intron", 
                    "descr":"posttranscriptional splicing involving exogenous GTP" },
"group2intron": {"name":"group II intron ribozyme", 
                    "hierarchy":"function > nazyme > group2intron", 
                    "descr":"posttranscriptional splicing involving lariat formation" },
"glmsribozyme": {"name":"glmS ribozyme", 
                    "hierarchy":"function > nazyme > glmsribozyme", 
                    "descr":"glucosamine-6-phosphate ribozyme, self-cleaves in presence of GlcN6P" },
"vsribozyme": {"name":"Varkud satellite ribozyme", 
                    "hierarchy":"function > nazyme > vsribozyme", 
                    "descr":"self-splicing component of mitochondrial satellite RNA in Neurospora" },
"ribonucleasep": {"name":"ribonuclease P ribozyme", 
                    "hierarchy":"function > nazyme > ribonucleasep", 
                    "descr":"processes RNA transcripts, e.g. removes precursor tRNA 5&prime; leader" },

"hairpin": {"name":"hairpin ribozyme", 
                    "hierarchy":"function > nazyme > hairpin", 
                    "descr":"simple hairpin ribozyme, processes rolling circle RNA" },
"hammerhead": {"name":"hammerhead ribozyme", 
                    "hierarchy":"function > nazyme > hammerhead", 
                    "descr":"hammerhead shaped ribozyme, processes rolling circle RNA" },
"hdvribozyme": {"name":"Hepatitis delta virus ribozyme", 
                    "hierarchy":"function > nazyme > hdvribozyme", 
                    "descr":"ribozyme encoded in virus genome, required for replication" },
"selfreplicating": {"name":"self-replicating", 
                    "hierarchy":"function > nazyme > selfreplicating", 
                    "descr":"nucleic acid able to copy itself or perform primer extension (model for origin of life)" },

"virusna": {"name":"virus NA", 
                    "hierarchy":"function > virusna", 
                    "descr":"nucleic acid associated with virus infectivity or replication" },

"telomere": {"name":"telomeric DNA", 
                    "hierarchy":"function > telomere", 
                    "descr":"telomeric DNA" },

"otherna": {"name":"other/unknown", 
                    "hierarchy":"function > otherna", 
                    "descr":"characterized non-coding nucleic acid with unassigned biological function" },


"olerna": {"name":"OLE RNA",
                    "hierarchy":"function > otherna > olerna", 
                    "descr":"highly conserved bacterial <u>O</u>rnate <u>L</u>arge <u>E</u>xtremophilic RNA, ~610 nt" },


"arrpofrna": {"name":"ARRPOF RNA", 
                    "hierarchy":"function > otherna > arrpofrna", 
                    "descr":"<u>A</u>rea <u>R</u>equired for <u>R</u>replication in a <u>P</u>lasmid of <u>F</u>usobacterium, ~255 nt" },

"roolrna": {"name":"ROOL RNA", 
                    "hierarchy":"function > otherna > roolrna", 
                    "descr":"<u>R</u>umen <u>O</u>riented, <u>O</u>rnate, <u>L</u>arge, prophage associated, ~580 nt" },

"golldrna": {"name":"GOLLD RNA",
                    "hierarchy":"function > otherna > golldrna", 
                    "descr":"<u>G</u>iant, <u>O</u>rnate, <u>L</u>ake and <u>L</u>actobacillales-derived, prophage associated, ~700 nt" },

"raiarna": {"name":"raiA RNA", 
                    "hierarchy":"function > otherna > raiarna", 
                    "descr":"abundant non-coding RNA family in Firmicutes, Actinobacteria, Gram-positive bacteria, ~250 nt" },

"nastructure": {"name":"structure", 
                    "hierarchy":"nastructure", 
                    "descr":"structural classification of nucleic acid" },
"double": {"name":"double helix", 
                    "hierarchy":"nastructure > double", 
                    "descr":"antiparallel double helix" },
"aform": {"name":"A-form double helix", 
                    "hierarchy":"nastructure > double > aform", 
                    "descr":"antiparallel double helix, right-handed, base-pairs tilted" },
"bform": {"name":"B-form double helix", 
                    "hierarchy":"nastructure > double > bform", 
                    "descr":"antiparallel double helix, right-handed, base-pairs perpendicular" },
"zform": {"name":"Z-form double helix", 
                    "hierarchy":"nastructure > double > zform", 
                    "descr":"antiparallel double helix, left-handed, zig-zag twist" },
"parallel": {"name":"parallel helix", 
                    "hierarchy":"nastructure > parallel", 
                    "descr":"two strands forming a parallel helix" },
"imotif": {"name":"I-motif", 
                    "hierarchy":"nastructure > parallel > imotif", 
                    "descr":"four strands forming two intercalated parallel duplexes" },
"triple": {"name":"triple helix", 
                    "hierarchy":"nastructure > triple", 
                    "descr":"helix composed of base triple steps" },
"quadruplex": {"name":"quadruplex", 
                    "hierarchy":"nastructure > quadruplex", 
                    "descr":"helix composed of base quadruple steps" },
"multiplex": {"name":"multiplex", 
                    "hierarchy":"nastructure > multiplex", 
                    "descr":"helix composed of steps with 5 or more hydrogen-bonded bases in plane" },
"holliday": {"name":"Holliday junction", 
                    "hierarchy":"nastructure > holliday", 
                    "descr":"two DNA helices forming a four-way junction" },

"feature": {"name":"feature", 
                    "hierarchy":"nastructure > feature", 
                    "descr":"motif or feature present within the assembly" },
"founding": {"name":"founding structure", 
                    "hierarchy":"nastructure > feature > founding", 
                    "descr":"historical di- or tri-nucleotide X-ray structure, not in PDB" },
"dodecamer": {"name":"DNA dodecamer", 
                    "hierarchy":"nastructure > feature > dodecamer", 
                    "descr":"DNA dodecamer duplex" },
"cyclic": {"name":"cyclic", 
                    "hierarchy":"nastructure > feature > cyclic", 
                    "descr":"cyclic oligonucleotide, e.g. cyclic AAAA" },
"bulge": {"name":"bulge", 
                    "hierarchy":"nastructure > feature > bulge", 
                    "descr":"one or more nucleotide residues excluded from basepairing in short DNA duplex" },
"mispair": {"name":"mispair", 
                    "hierarchy":"nastructure > feature > mispair", 
                    "descr":"non-Watson-Crick basepairs in short DNA duplex" },
"mispairaa": {"name":"A-A mispair", 
                    "hierarchy":"nastructure > feature > mispair > mispairaa", 
                    "descr":"A-A basepair in short DNA duplex" },
"mispairac": {"name":"A-C mispair", 
                    "hierarchy":"nastructure > feature > mispair > mispairac", 
                    "descr":"A-C basepair in short DNA duplex" },
"mispairag": {"name":"A-G mispair", 
                    "hierarchy":"nastructure > feature > mispair > mispairag", 
                    "descr":"A-G basepair in short DNA duplex" },
"mispaircc": {"name":"C-C mispair", 
                    "hierarchy":"nastructure > feature > mispair > mispaircc", 
                    "descr":"C-C basepair in short DNA duplex" },
"mispairct": {"name":"C-T mispair", 
                    "hierarchy":"nastructure > feature > mispair > mispairct", 
                    "descr":"C-T basepair in short DNA duplex" },
"mispairgg": {"name":"G-G mispair", 
                    "hierarchy":"nastructure > feature > mispair > mispairgg", 
                    "descr":"G-G basepair in short DNA duplex" },
"mispairgt": {"name":"G-T mispair", 
                    "hierarchy":"nastructure > feature > mispair > mispairgt", 
                    "descr":"G-T basepair in short DNA duplex" },
"mispairtt": {"name":"T-T mispair", 
                    "hierarchy":"nastructure > feature > mispair > mispairtt", 
                    "descr":"T-T basepair in short DNA duplex" },
"kinkturn": {"name":"kink turn",  /* not shown in tree */
                    "hierarchy":"nastructure > feature > kinkturn", 
                    "descr":"" }, 
"tetraloop": {"name":"tetraloop", /* not shown in tree */
                    "hierarchy":"nastructure > feature > tetraloop", 
                    "descr":"" },

"ornate": {"name":"Ornate RNA",
                      "hierarchy": "nastructure > ornate",
                      "descr": "natural large non-coding RNA with ornate folding pattern" },

"ornatemulti": {"name":"Multimeric Ornate RNA",
                      "hierarchy": "nastructure > ornate > ornatemulti",
                      "descr": "natural large non-coding ornate RNA forming multimeric assemblies" },


"designed": {"name":"designed assembly", 
                    "hierarchy":"nastructure > designed", 
                    "descr":"synthetic assembly" }

};

