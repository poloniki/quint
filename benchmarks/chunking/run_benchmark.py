"""Graded semantic-chunking benchmark: easy (cross-domain) vs hard (same-domain)."""
import sys, json
from collections import defaultdict
import numpy as np, pysbd
sys.path.insert(0, ".")
from quint.chunking.streaming import stream_boundaries
from model2vec import StaticModel

seg = pysbd.Segmenter(language="en", clean=False)
corpus = json.load(open("benchmarks/chunking/corpus.json"))
byc = defaultdict(list)
for p in corpus: byc[p["cluster"]].append(p)
m = StaticModel.from_pretrained("minishlab/potion-base-8M")
def embed(s):
    v = np.asarray(m.encode(s), np.float32); return v/(np.linalg.norm(v)+1e-9)

def make_doc(passages):
    sents, seams = [], set()
    for p in passages:
        ps = seg.segment(p["text"])
        if sents: seams.add(len(sents))
        sents += ps
    return sents, seams

rng = np.random.default_rng(0)
clusters = list(byc)
hard = [make_doc(byc[c]) for c in clusters]                     # within-cluster: subtle seams
easy = []
for _ in range(8):
    ci = rng.choice(len(clusters), size=4, replace=False)
    easy.append(make_doc([byc[clusters[i]][int(rng.integers(0,3))] for i in ci]))

def f1(pred, true, tol=1):
    P=set(pred)
    if not P: return (0.,0.,0.)
    prec=sum(any(abs(p-t)<=tol for t in true) for p in P)/len(P)
    rec=sum(any(abs(t-p)<=tol for p in P) for t in true)/len(true)
    return (2*prec*rec/(prec+rec) if prec+rec else 0., prec, rec)
def singleton(pred,N):
    e=[0]+sorted(pred)+[N]; s=[e[i+1]-e[i] for i in range(len(e)-1)]; return s.count(1)/len(s)

def evaluate(docs, min_size, ths):
    # pre-embed once per doc
    EV=[(np.array([embed(s) for s in sents]), seams, len(sents)) for sents,seams in docs]
    best=(-1,)
    for th in ths:
        Fs=Ps=Rs=Ss=0
        for V,seams,N in EV:
            cuts=stream_boundaries(list(range(N)), lambda i: V[i], th, min_size)
            F,P,R=f1(cuts,seams); Fs+=F; Ps+=P; Rs+=R; Ss+=singleton(cuts,N)
        k=len(EV)
        if Fs/k>best[0]: best=(Fs/k,Ps/k,Rs/k,Ss/k,th)
    return best

ths=[round(x,2) for x in np.arange(0.0,0.40,0.02)]
print(f"{'tier':5} {'method':18} {'F1':>5} {'prec':>5} {'rec':>5} {'single':>7}  best_th")
for tier,docs in [("easy",easy),("hard",hard)]:
    for name,ms in [("V0 (min_size=1)",1),("improved (min2)",2)]:
        F,P,R,S,th=evaluate(docs,ms,ths)
        print(f"{tier:5} {name:18} {F:5.2f} {P:5.2f} {R:5.2f} {S*100:6.0f}%  th={th:+.2f}")

# real transcript (no ground truth -> qualitative)
tsents=seg.segment(open("data/podcast_transcript.txt").read())
TV=np.array([embed(s) for s in tsents])
for name,ms,th in [("V0",1,0.10),("improved",2,0.12)]:
    cuts=stream_boundaries(list(range(len(tsents))), lambda i: TV[i], th, ms)
    print(f"transcript {name:12}: {len(cuts)+1} chunks, singletons {singleton(cuts,len(tsents))*100:.0f}%")
