## Restricted Boltzmann Machine (RBM)
##
## Note that in order to simplify the code all vectors are 1) by
## default row vectors unless otherwise noted, and 2) all implemented
## as a matrix with nrow=1.
## -------------------------------------------------------------------

library(stats)

## Sigmoid function
sigm <- function(x) 1 / (1 + exp(-x))

##' Initialize the RBM.
##'
##' @param nvis Number of visible units.
##' @param nhid Number of hidden units.
##' @return A list representing the RBM model.
rbm_setup <- function(nvis=784, nhid=500) {
  list(
    nvis = nvis,
    nhid = nhid,
    w = matrix(
      runif(nvis * nhid,
            -4 * sqrt(6 / (nhid + nvis)),
            4 * sqrt(6 / (nhid + nvis))),
      nvis, nhid),
    vbias = matrix(0, 1, nvis),
    hbias = matrix(0, 1, nhid)
  )
}

##' Get hidden state sample probability given a visible state.
##'
##' @param rbm The RBM model.
##' @param v The visible units.  Could be either a row vector or a
##'   matrix, where each row contains a data sample.
##' @return A vector (or matrix) of probability to sample the hidden
##'   state.  If return is a matrix, each row contains a set of hidden
##'   states.
rbm_propup <- function(rbm, v) {
  sigm(sweep(v %*% rbm$w, 2, rbm$hbias, "+"))
}

##' Get visible state sample probability given a hidden state.
##'
##' @param rbm The RBM model.
##' @param h The hidden state.  Could be a matrix, where each row
##'   contains a data sample.
##' @return A vector (or matrix) of probability to sample the visible
##'   state.  If return is a matrix, each column contains a set of
##'   hidden states.
rbm_propdown <- function(rbm, h) {
  sigm(sweep(h %*% t(rbm$w), 2, rbm$vbias, "+"))
}

##' Infers the state of hidden units given visible units.
##'
##' Converting to binary values, i.e., sampling according to the
##' probability is important to create the information bottleneck (by
##' Hinton, I have no idea what it means).
##'
##' @param rbm The RBM model.
##' @param v The visible units.
##' @return A list, E is the expectation of p(h|v), and sample is the
##'   samples from the distribution.
rbm_sample_h_given_v <- function(rbm, v) {
  E <- rbm_propup(rbm, v)
  list(E=E, sample=apply(E, c(1, 2), function(e) rbinom(1, 1, e)))
}

##' Infers the state of visible units given hidden units.
##'
##' Probability is usually OK without converting to binary values.
##'
##' @param rbm The RBM model.
##' @param h The hidden units.
##' @return A list, E is the expectation of p(v|h), and sample is the
##'   samples from the distribution.
rbm_sample_v_given_h <- function(rbm, h) {
  E = rbm_propdown(rbm, h)
  list(E=E, sample=apply(E, c(1, 2), function(e) rbinom(1, 1, e)))
}

##' One step of Gibbs sampling starting from visible state.
##'
##' @param rbm The RBM model.
##' @param v0 The initial visible state.
##' @return A vector of visible states after one step of Gibbs.
rbm_gibbs_vhv <- function(rbm, v0) {
  h1 <- rbm_sample_h_given_v(rbm, v0)
  rbm_sample_v_given_h(rbm, h1$E)
}

##' One step of Gibbs sampling starting from hidden state.
##'
##' @param rbm The RBM model.
##' @param h0 The initial hidden state.
##' @return A vector of hidden states after one step of Gibbs.
rbm_gibbs_hvh <- function(rbm, h0) {
  v1 <- rbm_sample_v_given_h(rbm, h0)
  rbm_sample_h_given_v(rbm, v1$E)
}

##' Compute the free energy of a visible units.
##'
##' @param rbm The RBM model.
##' @param v The visible units.
##' @return A vector (might be length of 1) as the energy of v.
rbm_free_energy <- function(rbm, v) {
  e0 <- rbm$vbias %*% t(v)

  wx.b <- sweep(v %*% rbm$w, 2, rbm$hbias, "+")
  e1 <- rbind(rowSums(log(1 + exp(wx.b))))

  -e0 - e1
}

##' Approximation to the reconstruction error
##'
##' This calculates the cross entropy between original data and data
##' generated after k steps of Gibbs sampling.
##'
##' @param v0 The original data sample.
##' @param vk Data after k steps of Gibbs sampling.
rbm_reconstruction_cost <- function(v0, vk) {
  mean(rowSums(v0 * log(vk) + (1 - v0) * log(1 - vk)))
}

##' Stochastic approximation to the pseudo-likelihood
##'
##' @param rbm The RBM model.
##' @param v The original data.
rbm_pseudo_likelihood_cost <- function(rbm, v) {
  v0 <- round(v)
  e0 <- rbm_free_energy(rbm, v0)

  v1 <- v0
  v1[, 1] <- 1 - v1[, 1]
  e1 <- rbm_free_energy(rbm, v1)

  mean(rbm$nvis * log(sigm(e1 - e0)))
}

##' Train the RBM with CD-k or PCD-k.
##'
##' @param rbm The RBM model.
##' @param data The input data.
##' @param lr The learning rate.
##' @param persistent T to perform PCD-k, CD-k otherwise.
##' @param k number of Gibbs step in CD-k/PCD-k.
##' @param batch.size Mini batch size.
##' @return A trained RBM model.
rbm_train <- function(rbm, data, lr=0.1, max.epoch=100, persistent=T,
                      k=1, batch.size=10) {

  n <- nrow(data)
  batch.ind <- matrix(sample.int(n), ceiling(n / batch.size), batch.size)
  chain.end <- matrix(runif(rbm$nvis), 1, rbm$nvis)
  cost <- rep(0, nrow(batch.ind))

  for (epoch in 1:max.epoch) {
    for (i in seq_len(nrow(batch.ind))) {

      ind <- batch.ind[i,]

      ## The rbind is to make sure batch.data is a 1-row matrix (row
      ## vector) in case (ind == 1).
      batch.data <- rbind(data[ind,])

      ## By convention, w is the weight matrix.  w_{ij} is the weight
      ## from v_i to h_j.  And b is the bias for visible units, c the
      ## bias for hidden units.

      ## positive phase

      h0 <- rbm_sample_h_given_v(rbm, batch.data)

      dw.pos <- t(batch.data) %*% h0$E
      db.pos <- batch.data
      dc.pos <- h0$E

      ## negative phase

      if (persistent) {
        len <- batch.size * k
        chain.end <- Reduce(
          function(chain.end, dummy) rbm_gibbs_vhv(rbm, chain.end)$E,
          1:len, chain.end[nrow(chain.end),], accumulate=T)
        chain.end <- t(simplify2array(chain.end, higher=F))
        chain.end <- chain.end[seq.int(1, len, k),]
      } else {
        chain.end <- Reduce(
          function(chain.end, dummy) rbm_gibbs_vhv(rbm, chain.end)$E,
          1:k, batch.data)
      }

      hk <- rbm_sample_h_given_v(rbm, chain.end)

      dw.neg <- t(chain.end) %*% hk$E
      db.neg <- chain.end
      dc.neg <- hk$E

      ## Summarize the gradient

      d.w <- (dw.pos - dw.neg) / batch.size
      d.b <- rbind(colMeans(db.pos - db.neg))
      d.c <- rbind(colMeans(dc.pos - dc.neg))

      ## update parameters

      rbm$w <- rbm$w + lr * d.w
      rbm$vbias <- rbm$vbias + lr * d.b
      rbm$hbias <- rbm$hbias + lr * d.c

      ## monitor cost

      cost[i] <- ifelse(persistent,
                        rbm_pseudo_likelihood_cost(rbm, batch.data),
                        rbm_reconstruction_cost(batch.data, chain.end))
    }

    print(sprintf("Training epoch %d, cost is %.6f, ", epoch, mean(cost)))
  }

  rbm
}

## RBM implementation ends here.
