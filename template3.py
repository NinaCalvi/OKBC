from template import TemplateBaseClass
import pickle

class Template3(TemplateBaseClass):
    """
    Template: r~r' ^ e1 r' e2
    """
 
    def __init__(self,kb,base_model,use_hard_triple_scoring=True,load_table=None,dump_file=None):
        super().__init__()
        self.kb=kb
        self.base_model=base_model
        self.use_hard_triple_scoring=use_hard_triple_scoring

        if(load_table==None):
            print("Load table is None, so beginning process_data")
            self.process_data()
            print("Process_data done")
            print("Begin Build table")
            self.build_table()
            print("END Build table")
            print("Begin dump data")
            self.dump_data(dump_file)
            print("END dump table")
        else:
            self.load_table(load_table)

    def process_data(self):
        """
        maps (e1,e2) to all r in data
        stores unique e1_r for building table
        """
        self.dict_e1_e2={}
        self.unique_e1_r={}

        for facts in self.kb.facts:
            key=(facts[0],facts[2])
            if(key not in self.dict_e1_e2):
                self.dict_e1_e2[key]=[]
            self.dict_e1_e2[key].append(facts[1])

            if((facts[0],facts[1]) not in self.unique_e1_r):
                self.unique_e1_r[(facts[0],facts[1])]=len(self.unique_e1_r)

    def build_table(self):
        """
        a table for each unique (e1,r)
        """
        entities=len(self.kb.entity_map)
        self.table={}
        total_els = len(self.unique_e1_r.keys())
        ctr = 0
        for (e1,r) in self.unique_e1_r.keys():
            if ctr%250==0:
                print("Processed %d"%(ctr))
            score_dict={}
            for u in range(entities):
            	sc,be = self.compute_score((e1,r,u))
            	if(sc!=0):
            		score_dict[u] = (sc,be)

            self.table[(e1,r)]=score_dict
            ctr+=1            



    def dump_data(self,filename):
        dump_dict={}
        dump_dict['dict_e1_e2']=self.dict_e1_e2
        dump_dict['unique_e1_r']=self.unique_e1_r
        dump_dict['table']=self.table

        with open(filename,'wb')  as inputfile:
            pickle.dump(dump_dict,inputfile)
    
    def load_table(self,filename):
        with open(filename,"rb") as f:
            dump_dict=pickle.load(f)
        self.dict_e1_e2=dump_dict['dict_e1_e2']
        self.unique_e1_r=dump_dict['unique_e1_r']
        self.table=dump_dict['table']
    

    def compute_score(self,triple):
        '''
        Returns template score for given triple
        Iterates over all e1,r depending on flag of use_hard_triple_scoring
        '''

        assert (len(triple) == 3), "Triple must contain three elements"

        score=0
        best=-1
        e2=triple[2]
        e1=triple[0]

        if(self.use_hard_triple_scoring==False):
            relations=len(self.kb.relation_map)

            for r in range(relations):
                relation_simi = self.base_model.get_relation_similarity(r, triple[1])
                model_score=self.base_model.compute_score(e1,r,e2)
                if(score<relation_simi*model_score):
                    score=relation_simi*model_score
                    best=r

        else:
            key=(e1,e2)
            if(key not in self.dict_e1_e2):
                score=0
            else:
                for r in self.dict_e1_e2[key]:
                    relation_simi = self.base_model.get_relation_similarity(r, triple[1])
                    if(score<relation_simi):
                        score=relation_simi
                        best=r
        return (score,best)


    # def get_input(fact):

