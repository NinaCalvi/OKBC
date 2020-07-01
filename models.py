import pickle
import numpy as np
import logging
import utils

class TypedDM():

    def __init__(self,pickle_dump_file):
        logging.info("Creating Base Model")
        logging.info("Loading Model weights from {0}".format(pickle_dump_file))
        with open(pickle_dump_file, "rb") as f:
            self.dump = pickle.load(f)

        entity_similarity_re = np.matmul(self.dump['entity_real'],np.transpose(self.dump['entity_real']))
        logging.info("Calculated entity real similarity matrix")
        entity_similarity_type = np.matmul(self.dump['entity_type'],np.transpose(self.dump['entity_type']))
        logging.info("Calculated entity type similarity matrix")
        self.entity_similarity = np.multiply(entity_similarity_re,entity_similarity_type)
        logging.info("Calculated entity similarity matrix")

        rel_similarity_re = np.matmul(self.dump['rel_real'],np.transpose(self.dump['rel_real']))
        logging.info("Calculated relation real similarity matrix")
        head_rel_similarity_type = np.matmul(self.dump['head_rel_type'],np.transpose(self.dump['head_rel_type']))
        logging.info("Calculated head relation type similarity matrix")
        tail_rel_similarity_type = np.matmul(self.dump['tail_rel_type'],np.transpose(self.dump['tail_rel_type']))
        logging.info("Calculated type relation type similarity matrix")
        self.relation_similarity = np.multiply(np.multiply(rel_similarity_re,head_rel_similarity_type),tail_rel_similarity_type)
        logging.info("Calculated relation similarity matrix")
        logging.info("Created Object of Base Model")

    def get_entity_similarity(self,e1,e2):
        # entity_similarity=np.dot(self.dump['entity_real'][e1],self.dump['entity_real'][e2])
        # type_compatibility = np.dot(self.dump['entity_type'][e1], self.dump['entity_type'][e2])
        # return entity_similarity*type_compatibility
        return self.entity_similarity[e1,e2]

    def get_relation_similarity(self, r1, r2):
        # relation_similarity = np.dot(
        #     self.dump['rel_real'][r1], self.dump['rel_real'][r2])
        # type_compatibility_head = np.dot(
        #     self.dump['head_rel_type'][r1], self.dump['head_rel_type'][r2])
        # type_compatibility_tail = np.dot(
        #     self.dump['tail_rel_type'][r1], self.dump['tail_rel_type'][r2])
        # return relation_similarity*type_compatibility_head*type_compatibility_tail
        return self.relation_similarity[r1,r2]

    def get_entity_similarity_list(self,e1,lis):
        return np.take(self.entity_similarity[e1], lis)

    def get_relation_similarity_list(self,r1,lis):
        return np.take(self.relation_similarity[r1], lis)

    def compute_score(self,e1,r,e2):
        score=self.dump['entity_real'][e1]*self.dump['rel_real'][r]*self.dump['entity_real'][e2]
        head_type_compatibility = self.dump['entity_type'][e1]*self.dump['head_rel_type']
        tail_type_compatibility = self.dump['entity_type'][e1]*self.dump['tail_rel_type']
        score=np.sum(score)
        head_type_compatibility=np.sum(head_type_compatibility)
        tail_type_compatibility=np.sum(tail_type_compatibility)
        return utils.sigmoid(score)*utils.sigmoid(head_type_compatibility)*utils.sigmoid(tail_type_compatibility)

    def get_hadamard_product(self,r1,r2):
        v1 = self.dump['rel_real'][r1]
        v2 = self.dump['rel_real'][r2]
        relation_product =  (v1 * v2)
        if(np.linalg.norm(relation_product)!=0):
            relation_product /= np.linalg.norm(relation_product)

        v1 = self.dump['head_rel_type'][r1]
        v2 = self.dump['head_rel_type'][r2]
        type_compatibility_head_product = (v1 * v2)
        if(np.linalg.norm(type_compatibility_head_product)!=0):
            type_compatibility_head_product /= np.linalg.norm(type_compatibility_head_product)

        v1 = self.dump['tail_rel_type'][r1]
        v2 = self.dump['tail_rel_type'][r2]
        type_compatibility_tail_product = (v1 * v2)
        if(np.linalg.norm(type_compatibility_tail_product)!=0):
            type_compatibility_tail_product /= np.linalg.norm(type_compatibility_tail_product)

        return [relation_product, type_compatibility_head_product, type_compatibility_tail_product]

    def get_relation_similarity_from_embedding(self,r1_emb, r2_emb):
        sim = 1.0
        for i,j in zip(r1_emb, r2_emb):
            sim = sim*np.dot(i,j)
        #if(sim>=0):
        #    return pow(sim,1/3)
        #return -pow(-sim,1/3)
        return (sim)

    def get_relation_embedding(self,r):
        return [self.dump['rel_real'][r], self.dump['head_rel_type'][r], self.dump['tail_rel_type'][r]]


class DM():

    def __init__(self,pickle_dump_file):
        logging.info("Creating Base Model")
        logging.info("Loading Model weights from {0}".format(pickle_dump_file))
        with open(pickle_dump_file, "rb") as f:
            self.dump = pickle.load(f)

        self.entity_similarity = np.matmul(self.dump['ent.weight'],np.transpose(self.dump['ent.weight']))
        logging.info("Calculated entity similarity matrix")
        self.rel_similarity = np.matmul(self.dump['rel.weight'],np.transpose(self.dump['rel.weight']))
        logging.info("Calculated relation similarity matrix")
        logging.info("Created Object of Base Model")

    def get_entity_similarity(self,e1,e2):
        # entity_similarity=np.dot(self.dump['entity_real'][e1],self.dump['entity_real'][e2])
        # type_compatibility = np.dot(self.dump['entity_type'][e1], self.dump['entity_type'][e2])
        # return entity_similarity*type_compatibility
        return self.entity_similarity[e1,e2]

    def get_relation_similarity(self, r1, r2):
        # relation_similarity = np.dot(
        #     self.dump['rel_real'][r1], self.dump['rel_real'][r2])
        # type_compatibility_head = np.dot(
        #     self.dump['head_rel_type'][r1], self.dump['head_rel_type'][r2])
        # type_compatibility_tail = np.dot(
        #     self.dump['tail_rel_type'][r1], self.dump['tail_rel_type'][r2])
        # return relation_similarity*type_compatibility_head*type_compatibility_tail
        return self.relation_similarity[r1,r2]

    def get_entity_similarity_list(self,e1,lis):
        return np.take(self.entity_similarity[e1], lis)

    def get_relation_similarity_list(self,r1,lis):
        return np.take(self.relation_similarity[r1], lis)

    def compute_score(self,e1,r,e2):
        score=self.dump['ent.weight'][e1]*self.dump['rel.weight'][r]*self.dump['ent.weight'][e2]
        score=np.sum(score)
        return utils.sigmoid(score)

    def get_hadamard_product(self,r1,r2):
        v1 = self.dump['rel.weight'][r1]
        v2 = self.dump['rel.weight'][r2]
        relation_product =  (v1 * v2)
        if(np.linalg.norm(relation_product)!=0):
            relation_product /= np.linalg.norm(relation_product)

        return relation_product

    def get_relation_similarity_from_embedding(self,r1_emb, r2_emb):
        sim = 1.0
        for i,j in zip(r1_emb, r2_emb):
            sim = sim*np.dot(i,j)
        #if(sim>=0):
        #    return pow(sim,1/3)
        #return -pow(-sim,1/3)
        return (sim)

    def get_relation_embedding(self,r):
        return [self.dump['rel.weight'][r]]





class TriVec():

    def init(self, pickle_dump_file):
        with open(pickle_dump_file, "rb") as f:
            self.dump = pickle.load(f)

        # emb = np.concatenate((self.dump['Ee1'],self.dump['Ee2'],self.dump['Ee3']), axis=1)
        # rel = np.concatenate((self.dump['Er1'],self.dump['Er2'],self.dump['Er3']), axis=1)

        self.entity_similarity = np.matmul(self.dump['ents'], np.transpose(self.dump['ents']))
        logging.info("Calculated entity similarity matrix")
        self.rel_similarity = np.matmul(self.dump['rels'], np.transpose(self.dump['rels']))
        logging.info("Calculated relation similarity matrix")
        logging.info("Created Object of Base Model")



    def get_entity_similarity(self, e1, e2):
        return self.entity_similarity[e1,e2]

    def get_relation_similarity(self, r1, r2):
        return self.relation_similarity[r1,r2]

    def compute_score(self, e1, r, e2):

        s = self.dump['ents'][e1]
        rel = self.dump['rels'][r]
        o = self.dump['ents'][e2]

        sub_em_v1, sub_em_v2, sub_em_v3 = np.split(s, 3, axis=1)
        rel_em_v1, rel_em_v2, rel_em_v3 = np.split(rel, 3, axis=1)
        obj_em_v1, obj_em_v2, obj_em_v3 = np.split(o, 3, axis=1)

        em_interaction = sub_em_v1*rel_em_v1*obj_em_v3 \
                         + sub_em_v2*rel_em_v2*obj_em_v2 \
                         + sub_em_v3*rel_em_v3*obj_em_v1

        score = np.sum(em_interaction)
        return utils.sigmoid(score)






class TypedComplex():

    def init(self, pickle_dump_file):
        with open(pickle_dump_file, "rb") as f:
            self.dump = pickle.load(f)
        raise "Not Implemented"

    def get_entity_similarity(self, e1, e2):
        entity_similarity = np.dot(self.dump['entity_real'][e1], self.dump['entity_real'][e2])+np.dot(
            self.dump['entity_im'][e1], self.dump['entity_im'][e2])
        type_compatibility = np.dot(
            self.dump['entity_type'][e1], self.dump['entity_type'][e2])
        return entity_similarity*type_compatibility

    def get_relation_similarity(self, r1, r2):
        relation_similarity = np.dot(self.dump['rel_real'][r1], self.dump['rel_real'][r2])+np.dot(
            self.dump['rel_im'][r1], self.dump['rel_im'][r2])
        type_compatibility_head = np.dot(
            self.dump['head_rel_type'][r1], self.dump['head_rel_type'][r2])
        type_compatibility_tail = np.dot(
            self.dump['tail_rel_type'][r1], self.dump['tail_rel_type'][r2])
        return relation_similarity*type_compatibility_head*type_compatibility_tail

    def compute_score(self, e1, r, e2):
        s_re = self.dump['entity_real'][e1]
        s_im = self.dump['entity_im'][e1]
        o_re = self.dump['entity_real'][e2]
        o_im = self.dump['entity_im'][e2]
        r_re = self.dump['rel_real'][r]
        r_im = self.dump['rel_im'][r]

        score = (s_re*o_re+s_im*o_im)*r_re + (s_re*o_im-s_im*o_re)*r_im
        head_type_compatibility = self.dump['entity_type'][e1] * \
            self.dump['head_rel_type']
        tail_type_compatibility = self.dump['entity_type'][e1] * \
            self.dump['tail_rel_type']
        score = np.sum(score)
        head_type_compatibility = np.sum(head_type_compatibility)
        tail_type_compatibility = np.sum(tail_type_compatibility)
        return utils.sigmoid(score)*utils.sigmoid(head_type_compatibility)*utils.sigmoid(tail_type_compatibility)
