# Shared-LoRA-Subspaces-for-almost-Strict-Continual-Learning

### Step by Step working : 
---

# Share: Step by Step Working with Dimensions

## Setup — Define Everything First

```
Pretrained model weight matrix:
W0 : (768 × 768)    ← one layer of RoBERTa base

LoRA rank:
r = 32

Share hyperparameters:
k = 32    ← number of principal basis vectors
p = 8     ← pseudo rank of coefficients
φ = 4     ← temporary basis vectors for adaptation

So LoRA matrices per task:
B : (768 × 32)
A : (32 × 768)
```

---

## Step 1 — Initialization (2 tasks available)

### Stack B matrices from 2 existing LoRA adapters

```
B1 : (768 × 32)    ← adapter from Task 1
B2 : (768 × 32)    ← adapter from Task 2

Stack column wise:
D_B = [B1, B2] : (768 × 64)
```

### Center the matrix

```
mean_B = mean of D_B across columns : (768 × 1)
D_B_centered = D_B - mean_B         : (768 × 64)
```

### Run SVD

```
SVD(D_B_centered) = U · Σ · Vᵀ

U : (768 × 64)    ← left singular vectors
Σ : (64 × 64)     ← singular values
Vᵀ: (64 × 64)     ← right singular vectors
```

### Extract top k principal basis vectors

```
β = U[:, 1:k] : (768 × 32)    ← frozen forever
```

### Do exactly same for A matrices

```
A1 : (32 × 768)
A2 : (32 × 768)

D_A = [A1, A2] : (64 × 768)   ← stacked differently

SVD → extract top k vectors:
α = V[:, 1:k] : (768 × 32)    ← frozen forever
```

### Initialize task coefficients analytically

```
For Task 1:
ϵ_β_1 = βᵀ · B1
       : (32 × 768) · (768 × 32) = (32 × 32)
       but we use p=8 so → (32 × 8)

ϵ_α_1 = αᵀ · A1ᵀ
       : (32 × 768) · (768 × 32) = (32 × 32)
       but we use p=8 so → (32 × 8)

Same for Task 2:
ϵ_β_2 : (32 × 8)
ϵ_α_2 : (32 × 8)
```

### What we have after initialization

```
Frozen (stored once):
β : (768 × 32)
α : (768 × 32)

Per task (stored per task):
ϵ_β_1 : (32 × 8)
ϵ_α_1 : (32 × 8)
ϵ_β_2 : (32 × 8)
ϵ_α_2 : (32 × 8)

Forward pass for Task 1:
h = W0x + (β · ϵ_β_1)(α · ϵ_α_1)ᵀ x
        : (768×32)·(32×8) = (768×8)
  (α · ϵ_α_1)ᵀ : (8×768)
  (768×8)·(8×768) = (768×768)   ← same shape as W0
```

---

## Step 2 — Task 3 Arrives (Scenario A — Adapter Available)

### New adapter arrives

```
B3 : (768 × 32)
A3 : (32 × 768)
```

### Project directly onto existing basis

```
ϵ_β_3 = βᵀ · B3
       : (32 × 768) · (768 × 32) = (32 × 32)
       truncate to p=8 → (32 × 8)

ϵ_α_3 = αᵀ · A3ᵀ
       : (32 × 768) · (768 × 32) = (32 × 32)
       truncate to p=8 → (32 × 8)
```

### No training needed — go straight to merging

---

## Step 3 — Task 4 Arrives (Scenario B — Only Data Available)

### Take subset of existing basis as temporary basis

```
β_temp = β[:, 1:φ] : (768 × 4)    ← top 4 columns of β
α_temp = α[:, 1:φ] : (768 × 4)    ← top 4 columns of α
```

### Initialize random coefficients

```
ϵ_β_temp : (4 × 8)    ← randomly initialized
ϵ_α_temp : (4 × 8)    ← randomly initialized
```

### Temporary forward pass during training

```
B_temp = β_temp · ϵ_β_temp
       : (768 × 4) · (4 × 8) = (768 × 8)

A_temp = (α_temp · ϵ_α_temp)ᵀ
       : (768 × 4) · (4 × 8) = (768 × 8) → transpose → (8 × 768)

h = W0x + B_temp · A_temp · x
  : (768×8) · (8×768) = (768×768)   ← same shape as W0
```

### Train only β_temp and ϵ_β_temp, ϵ_α_temp

```
Trainable parameters:
β_temp   : (768 × 4)
α_temp   : (768 × 4)
ϵ_β_temp : (4 × 8)
ϵ_α_temp : (4 × 8)

Total = 768×4 + 768×4 + 4×8 + 4×8
      = 3072 + 3072 + 32 + 32
      = 6208 parameters

Compare to full LoRA:
768×32 + 32×768 = 24576 + 24576 = 49152 parameters

Share trains 8× fewer parameters temporarily
```

---

## Step 4 — Merging After Task 4

### Reconstruct all previous task adapters

```
B̂1 = β · ϵ_β_1 : (768×32)·(32×8) = (768×8)
B̂2 = β · ϵ_β_2 : (768×32)·(32×8) = (768×8)
B̂3 = β · ϵ_β_3 : (768×32)·(32×8) = (768×8)
B̂4 = β_temp · ϵ_β_temp : (768×4)·(4×8) = (768×8)
```

### Stack all reconstructed adapters

```
D_B_new = [B̂1, B̂2, B̂3, B̂4] : (768 × 32)
          4 tasks × p=8 columns each
```

### Run SVD to get updated basis

```
SVD(D_B_new) = U · Σ · Vᵀ

Extract top k=32 vectors:
β_new = U[:, 1:32] : (768 × 32)    ← updated frozen basis
```

### Analytically reproject all task coefficients

```
ϵ_β_1_new = β_newᵀ · B̂1
           : (32×768) · (768×8) = (32×8)

ϵ_β_2_new = β_newᵀ · B̂2
           : (32×768) · (768×8) = (32×8)

ϵ_β_3_new = β_newᵀ · B̂3
           : (32×768) · (768×8) = (32×8)

ϵ_β_4_new = β_newᵀ · B̂4
           : (32×768) · (768×8) = (32×8)

Same process repeated for α side
```

### Discard temporary parameters

```
Discard β_temp, α_temp, ϵ_β_temp, ϵ_α_temp
These were only needed during adaptation
```

---

## Step 5 — Final State After 4 Tasks

### What is stored

```
Frozen basis (stored once for all tasks):
β : (768 × 32)    = 24,576 values
α : (768 × 32)    = 24,576 values

Per task coefficients:
ϵ_β_1 : (32 × 8) = 256 values
ϵ_α_1 : (32 × 8) = 256 values
ϵ_β_2 : (32 × 8) = 256 values
ϵ_α_2 : (32 × 8) = 256 values
ϵ_β_3 : (32 × 8) = 256 values
ϵ_α_3 : (32 × 8) = 256 values
ϵ_β_4 : (32 × 8) = 256 values
ϵ_α_4 : (32 × 8) = 256 values
```

### Compare with vanilla LoRA

```
LoRA for 4 tasks:
4 × (B + A) = 4 × (768×32 + 32×768)
            = 4 × 49,152
            = 196,608 values

Share for 4 tasks:
Basis + Coefficients
= (24,576 + 24,576) + 4×(256+256+256+256)
= 49,152 + 4,096
= 53,248 values

Savings = 196,608 / 53,248 ≈ 3.7× for 4 tasks
Savings grow dramatically as T increases
```

---

## Step 6 — Inference for Any Task

```
To run Task 2:
Load β  : (768 × 32)   ← always in memory
Load α  : (768 × 32)   ← always in memory
Swap in ϵ_β_2 : (32 × 8)   ← tiny swap
Swap in ϵ_α_2 : (32 × 8)   ← tiny swap

Forward pass:
h = W0x + (β · ϵ_β_2)(α · ϵ_α_2)ᵀ x

To switch to Task 3:
Only swap ϵ_β_3, ϵ_α_3   ← 512 values total
Everything else stays same
```

---

## Summary of All Dimensions

```
Matrix          Shape         Frozen?    Per task?
─────────────────────────────────────────────────
W0              768 × 768     Yes        No
β               768 × 32      Yes        No
α               768 × 32      Yes        No
ϵ_β             32  × 8       No         Yes
ϵ_α             32  × 8       No         Yes
β_temp          768 × 4       No         Temporary
ϵ_β_temp        4   × 8       No         Temporary
─────────────────────────────────────────────────
Temporary parameters exist only during adaptation
Discarded after merging
```

Medium article : [Link](https://medium.com/@apurv.pujari1/what-if-ai-could-learn-like-humans-continuously-without-ever-forgetting-well-almost-2571b47b21d5)

Paper : [Link](https://arxiv.org/abs/2602.06043)
