import scipy.stats as scs
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def compute_posterior_laws(results_a, results_b,
                           prior_alpha_a=1, prior_beta_a=1, 
                           prior_alpha_b=1, prior_beta_b=1):
    
    posterior_alpha_a = prior_alpha_a + np.sum(results_a)
    posterior_beta_a = prior_beta_a + len(results_a) - np.sum(results_a)

    posterior_alpha_b = prior_alpha_b + np.sum(results_b)
    posterior_beta_b = prior_beta_b + len(results_b) - np.sum(results_b)
    
    posterior_law_a = scs.beta(posterior_alpha_a, posterior_beta_a)
    posterior_law_b = scs.beta(posterior_alpha_b, posterior_beta_b)
    
    return posterior_law_a, posterior_law_b


def compute_posterior_probs(posterior_law_a, posterior_law_b, steps=100):
    
    posterior_a = []
    posterior_b = []
    
    for i in range(steps):
        inf_ = i * (1 / (steps))
        sup_ = (i + 1) * (1 / (steps))
        posterior_a.append(posterior_law_a.cdf(sup_) - posterior_law_a.cdf(inf_))
        posterior_b.append(posterior_law_b.cdf(sup_) - posterior_law_b.cdf(inf_))
    
    return posterior_a, posterior_b


def compute_joint_posterior_probs(posterior_a, posterior_b, steps=100):
    
    posterior_joint = np.zeros((steps, steps))
    
    for i in range(steps):
        for j in range(steps):
            posterior_joint[i, j] = posterior_a[i] * posterior_b[j]
            
    return posterior_joint


def compute_error_probability(posterior_joint_probs, steps):
        
    proba_b_sup_a = 0
    for i in range(steps):
        for j in range(i + 1, steps):
            proba_b_sup_a += posterior_joint_probs[i, j]
            
    proba_a_sup_b = 0
    for i in range(steps):
        for j in range(0, i):
            proba_a_sup_b += posterior_joint_probs[i, j]
            
    proba_a_equal_b = 0
    for i in range(steps):
        proba_a_equal_b += posterior_joint_probs[i, i]
    
    return proba_b_sup_a, proba_a_sup_b, proba_a_equal_b

def compute_error_probability_from_results(results_a, results_b, steps=100):
    posterior_law_a, posterior_law_b = compute_posterior_laws(results_a, results_b)
    posterior_a, posterior_b = compute_posterior_probs(posterior_law_a, posterior_law_b, steps=100)
    posterior_joint_probs = compute_joint_posterior_probs(posterior_a, posterior_b)
    return compute_error_probability(posterior_joint_probs, steps)


def punctual_loss(i, j, var, steps):
    if var == "A":
        return max(j * (1 / steps) - i * (1/steps), 0)

    if var == "B":
        return max(i * (1 / steps) - j * (1/steps), 0)
    

def compute_loss(posterior_joint_probs, var, steps):
    
    loss = 0
    for i in range(steps):
        for j in range(steps):
            loss += posterior_joint_probs[i, j] * punctual_loss(i, j, var, steps)
    
    return loss