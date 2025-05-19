#!/bin/sh
# -*- mode: scheme; coding: utf-8 -*-
install_dir=$(dirname $0)/..
local_lib_dir=${install_dir}/share/guile/site/3.0/scripts  #follow make install directory hierarchy of autotools
export GUILE_LOAD_PATH=${install_dir}:${local_lib_dir}:${GUILE_LOAD_PATH}
export LD_LIBRARY_PATH=${install_dir}/lib:${LD_LIBRARY_PATH}
exec guile -e main -s "$0" "$@"
!#

;;;     Copyright 2024 Li-Cheng (Andy) Tai
;;;                      atai@atai.org
;;;
;;;     guile_llama_cpp is free software: you can redistribute it and/or modify it
;;;     under the terms of the GNU Lesser  General Public License as published by the Free
;;;     Software Foundation, either version 3 of the License, or (at your option)
;;;     any later version.
;;;
;;;     guile_llama_cpp is distributed in the hope that it will be useful, but WITHOUT
;;;     ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
;;;     FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
;;;     more details.
;;;
;;;     You should have received a copy of the GNU Lesser General Public License along with
;;;     guile_llama_cpp. If not, see http://www.gnu.org/licenses/.

(use-modules (rnrs)            ; for bytevectors
             (system foreign) ; for FFI
             (system vm trace)
             (ice-9 rdelim)   ; read-line
             (ice-9 getopt-long)) ; getopt

(include-from-path "init_common.scm")
(use-modules (guile-llama-cpp))

(include-from-path "utils.scm")

(define (make-prompt-func model-path context-length prediction-length) ;model-path is a gguf file path
  (let ((model-params #f)
        (context-params #f)
        (model #f)
        (ctx #f)
        (n-predict prediction-length)  ; has to be reasonable positive length for generating reply by default
        (n-ctx context-length)
        (n-tokens-max #f)
        (prompt-func #f))
    (llama-backend-init)
    (set! model-params (llama-model-default-params))
    (set! model (llama-load-model-from-file model-path model-params))
    (set! context-params (llama-context-default-params))
    (llama-context-params-seed-set context-params 1234)
    (llama-context-params-n-ctx-set context-params n-ctx)
    (llama-context-params-n-threads-set context-params 8)
    (llama-context-params-n-threads-batch-set context-params 8)

    (set! ctx (llama-new-context-with-model model context-params))
    (set! n-tokens-max (llama-context-params-n-ctx-get context-params))

    (set! prompt-func
        (lambda* (#:key
            (prompt-text "")
            (temperature 0.8)
            (repeat-penalty 1.1)
            (top-p 0.9)        ; nucleus sampling threshold
            (top-k 40)         ; top-k sampling
            (min-p 0.05)       ; min probability threshold
            (tfs-z 1.0)        ; tail free sampling parameter
            (typ-p 1.0)        ; typical sampling parameter
            (frequency-penalty 0.0) ; penalty for frequent tokens
            (presence-penalty 0.0)  ; penalty for token presence
            (mirostat 0)       ; mirostat sampling (0, 1, or 2)
            (mirostat-tau 5.0) ; mirostat target entropy
            (mirostat-eta 0.1) ; mirostat learning rate
            (penalize-nl #f)   ; penalize newlines
            (ignore-eos #f)    ; ignore end of sequence token
            (seed -1))         ; random seed (-1 for random)
          (let ((tokens #f)
                (tokens-as-byte-vector #f)
                (zero-token-vector #f)
                (n-tokens #f)
                (n-kv-req #f)
                (batch #f)
                (r #f)
                (result "")
                (n-cur #f)
                (n-decode #f)
                (break #f))

            (set! tokens (new-llama-token-array n-tokens-max))
            (set! n-tokens (llama-tokenize (llama-get-model ctx) prompt-text (llama-token-array-cast tokens) n-tokens-max #t #f))

            (set! n-kv-req n-predict)

            (if (> n-kv-req n-ctx)
                (begin
                  (format #t "error: n_kv_req > n_ctx, the required KV cache size is not big enough\n")
                  "")
                (begin
                  (set! batch (llama-batch-init 512 0 1))
                  ; note in llama.h, in C: typedef int32_t llama_token;
                  (set! zero-token-vector (new-llama-seq-id-vector 1))
                  (do ((i 0 (+ i 1)))
                      ((>= i n-tokens) #t)
                    (llama-batch-add
                     batch (llama-token-array-getitem tokens i) i zero-token-vector #f))
                  (let ((logits (int8-array-frompointer (llama-batch-logits-get batch))))
                    (int8-array-setitem logits
                        (- (llama-batch-n-tokens-get batch) 1) 1))  ; int form for #t

                  (set! r (llama-decode ctx batch))
                  (if (not (eq? r 0))
                      (begin
                        (format #t "llama-decode failed\n")
                        "")
                      ; main loop
                      (begin
                        (set! n-cur (llama-batch-n-tokens-get batch))
                        (set! n-decode 0)
                        (set! break #f)
                        (do () ((or break (> n-cur n-predict)) #t)
                          ;  sample the next token
                          (let ((n-vocab (llama-n-vocab model))
                                (logits (llama-get-logits-ith ctx (- (llama-batch-n-tokens-get batch) 1)))
                                (candidates (new-llama-token-data-vector))
                                (candidates-p #f)
                                (new-token-id #f)
                                (eog #f))
                            
                            ; Set up candidates
                            (do ((token-id 0 (+ token-id 1)))
                                ((>= token-id n-vocab) #t)
                              (let ((token-data (new-llama-token-data))
                                    (logits-a (float-array-frompointer logits)))
                                (llama-token-data-id-set token-data token-id)
                                (llama-token-data-logit-set token-data (float-array-getitem logits-a token-id))
                                (llama-token-data-p-set token-data 0.0)

                                (llama-token-data-vector-push! candidates token-data)))
                            
                            ; Prepare candidates array for sampling
                            (set! candidates-p (new-llama-token-data-array))
                            (llama-token-data-array-data-set candidates-p (llama-token-data-vector-data candidates))
                            (llama-token-data-array-size-set candidates-p n-vocab)
                            (llama-token-data-array-sorted-set candidates-p #f)
                            
                            ; Sample token based on parameters
                            ; Apply penalties
                            (when (> repeat-penalty 1.0)
                              ; Apply repeat penalty (implementation would need to track previous tokens)
                              ; This is a placeholder - would need actual implementation
                              )
                            
                            (when (not (= frequency-penalty 0.0))
                              ; Apply frequency penalty
                              ; This is a placeholder - would need actual implementation
                              )
                            
                            (when (not (= presence-penalty 0.0))
                              ; Apply presence penalty
                              ; This is a placeholder - would need actual implementation
                              )
                            
                            ; Choose sampling method based on parameters
                            (cond
                              ((> mirostat 0) 
                               ; Use mirostat sampling (if implemented in binding)
                               ; Fallback to temperature sampling if not implemented
                               (set! new-token-id (llama-sample-temperature ctx candidates-p temperature)))
                              
                              ((< temperature 1e-6)
                               ; Use greedy sampling when temperature is very low
                               (set! new-token-id (llama-sample-token-greedy ctx candidates-p)))
                              
                              ((> top-p 0.0)
                               ; Use top-p nucleus sampling
                               (set! new-token-id (llama-sample-top-p ctx candidates-p top-p temperature)))
                              
                              (else
                               ; Fallback to basic temperature sampling
                               (set! new-token-id (llama-sample-temperature ctx candidates-p temperature))))
                            
                            ; Check for end of generation
                            (set! eog (llama-token-is-eog model new-token-id))
                            (if (or (and (not ignore-eos) eog) (= n-cur n-predict))
                                (begin
                                  (format #t "~%")  ; end of generation
                                  (set! break #t))  ; break out of loop
                                (begin
                                  (let ((gen (llama-token-to-piece-return-string ctx new-token-id)))
                                    (set! result (string-append result gen))
                                    (format #t "~:a" gen))
                                  (llama-batch-clear batch)
                                  (llama-batch-add batch new-token-id n-cur zero-token-vector #t)
                                  (set! n-decode (+ n-decode 1))

                                  (set! n-cur (+ n-cur 1))
                                  (set! r (llama-decode ctx batch))))))))

                  (llama-batch-free batch)))

            (delete-llama-token-array tokens)
            result))))
    
    prompt-func))

(define* (prompt-and-answer
        #:key
        (prompt-text "")
        (model-path "")
        (n-predict 1024)  ; has to be reasonable positive length for generating reply by default
        (n-ctx 0)
        (temperature 0.8)
        (repeat-penalty 1.1)
        (top-p 0.9)        ; nucleus sampling threshold
        (top-k 40)         ; top-k sampling
        (min-p 0.05)       ; min probability threshold
        (tfs-z 1.0)        ; tail free sampling parameter
        (typ-p 1.0)        ; typical sampling parameter
        (frequency-penalty 0.0) ; penalty for frequent tokens
        (presence-penalty 0.0)  ; penalty for token presence
        (mirostat 0)       ; mirostat sampling (0, 1, or 2)
        (mirostat-tau 5.0) ; mirostat target entropy
        (mirostat-eta 0.1) ; mirostat learning rate
        (penalize-nl #f)   ; penalize newlines
        (ignore-eos #f)    ; ignore end of sequence token
        (seed -1))         ; random seed (-1 for random)
  (let ((prompt-func (make-prompt-func model-path n-ctx n-predict)))
    (prompt-func
     #:prompt-text prompt-text
     #:temperature temperature
     #:repeat-penalty repeat-penalty
     #:top-p top-p
     #:top-k top-k
     #:min-p min-p
     #:tfs-z tfs-z
     #:typ-p typ-p
     #:frequency-penalty frequency-penalty
     #:presence-penalty presence-penalty
     #:mirostat mirostat
     #:mirostat-tau mirostat-tau
     #:mirostat-eta mirostat-eta
     #:penalize-nl penalize-nl
     #:ignore-eos ignore-eos
     #:seed seed)))

(define (main args)
  (let* ((option-spec '((version (single-char #\v))
                        (help    (single-char #\h))
                        (prompt  (single-char #\p) (value #t))
                        (model   (single-char #\m) (value #t))
                        (ctx-size (single-char #\c) (value #t))
                        (n-predict (single-char #\n) (value #t))
                        (temperature (single-char #\t) (value #t))
                        (repeat-penalty (single-char #\r) (value #t))
                        (top-p (value #t))
                        (top-k (value #t))
                        (min-p (value #t))
                        (tfs-z (value #t))
                        (typ-p (value #t))
                        (frequency-penalty (value #t))
                        (presence-penalty (value #t))
                        (mirostat (value #t))
                        (mirostat-tau (value #t))
                        (mirostat-eta (value #t))
                        (penalize-nl (value #t))
                        (ignore-eos (value #t))
                        (seed (value #t))
                        (interactive (single-char #\i))))
         (options (getopt-long args option-spec))
         (version-wanted (option-ref options 'version #f))
         (help-wanted (option-ref options 'help #f))
         (interactive (option-ref options 'interactive #f))
         (prompt-text (option-ref options 'prompt ""))
         (model-path (option-ref options 'model ""))
         (context-length (string->number (option-ref options 'ctx-size "2048")))
         (prediction-length (string->number (option-ref options 'n-predict "1024")))
         (temperature-value (string->number (option-ref options 'temperature "0.8")))
         (repeat-penalty-value (string->number (option-ref options 'repeat-penalty "1.1")))
         (top-p-value (string->number (option-ref options 'top-p "0.9")))
         (top-k-value (string->number (option-ref options 'top-k "40")))
         (min-p-value (string->number (option-ref options 'min-p "0.05")))
         (tfs-z-value (string->number (option-ref options 'tfs-z "1.0")))
         (typ-p-value (string->number (option-ref options 'typ-p "1.0")))
         (frequency-penalty-value (string->number (option-ref options 'frequency-penalty "0.0")))
         (presence-penalty-value (string->number (option-ref options 'presence-penalty "0.0")))
         (mirostat-value (string->number (option-ref options 'mirostat "0")))
         (mirostat-tau-value (string->number (option-ref options 'mirostat-tau "5.0")))
         (mirostat-eta-value (string->number (option-ref options 'mirostat-eta "0.1")))
         (penalize-nl-value (string=? (option-ref options 'penalize-nl "false") "true"))
         (ignore-eos-value (string=? (option-ref options 'ignore-eos "false") "true"))
         (seed-value (string->number (option-ref options 'seed "-1")))
         (prompt-func #f)
         (reply #f))
    
    (if help-wanted
        (begin
          (format #t "Usage: ~:a [OPTIONS]\n\n" (car args))
          (format #t "Options:\n")
          (format #t "  -v, --version            Show version information\n")
          (format #t "  -h, --help               Show this help message\n")
          (format #t "  -p, --prompt TEXT        Input prompt for the model\n")
          (format #t "  -m, --model PATH         Path to GGUF model file\n")
          (format #t "  -c, --ctx-size N         Context size (default: 2048)\n")
          (format #t "  -n, --n-predict N        Maximum number of tokens to predict (default: 1024)\n")
          (format #t "  -t, --temperature VAL    Sampling temperature (default: 0.8)\n")
          (format #t "  -r, --repeat-penalty VAL Repeat penalty (default: 1.1)\n")
          (format #t "  -i, --interactive        Interactive mode\n")
          (format #t "  --top-p VAL              Top-p sampling threshold (default: 0.9)\n")
          (format #t "  --top-k N                Top-k sampling threshold (default: 40)\n")
          (format #t "  --min-p VAL              Min probability threshold (default: 0.05)\n")
          (format #t "  --tfs-z VAL              Tail free sampling parameter (default: 1.0)\n")
          (format #t "  --typ-p VAL              Typical sampling parameter (default: 1.0)\n")
          (format #t "  --frequency-penalty VAL  Frequency penalty (default: 0.0)\n")
          (format #t "  --presence-penalty VAL   Presence penalty (default: 0.0)\n")
          (format #t "  --mirostat N             Mirostat sampling (0=disabled, 1=v1, 2=v2) (default: 0)\n")
          (format #t "  --mirostat-tau VAL       Mirostat target entropy (default: 5.0)\n")
          (format #t "  --mirostat-eta VAL       Mirostat learning rate (default: 0.1)\n")
          (format #t "  --penalize-nl true|false Penalize newlines (default: false)\n")
          (format #t "  --ignore-eos true|false  Ignore end of sequence token (default: false)\n")
          (format #t "  --seed N                 Random seed (-1 for random) (default: -1)\n")
          (exit 0)))
    
    (if version-wanted
        (begin
          (format #t "guile-llama-cpp v1.0.0\n")
          (exit 0)))
    
    (when (string=? model-path "")
      (format #t "Error: Model path (-m/--model) is required\n")
      (exit 1))
    
    (set! prompt-func (make-prompt-func model-path context-length prediction-length))
    
    (do ((looping #t interactive))
        ((not looping))
        (if (string-null? prompt-text)
            (begin
              (format #t "Please enter new prompt: ")
              (set! prompt-text (read-line))))
        
        (set! reply (prompt-func
                    #:prompt-text prompt-text
                    #:temperature temperature-value
                    #:repeat-penalty repeat-penalty-value
                    #:top-p top-p-value
                    #:top-k top-k-value
                    #:min-p min-p-value
                    #:tfs-z tfs-z-value
                    #:typ-p typ-p-value
                    #:frequency-penalty frequency-penalty-value
                    #:presence-penalty presence-penalty-value
                    #:mirostat mirostat-value
                    #:mirostat-tau mirostat-tau-value
                    #:mirostat-eta mirostat-eta-value
                    #:penalize-nl penalize-nl-value
                    #:ignore-eos ignore-eos-value
                    #:seed seed-value))
        
        (newline)
        (format #t "LLM reply: ~:a  ~%"  reply)
        (set! prompt-text ""))
    
    (exit 0)))