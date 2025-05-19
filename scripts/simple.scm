#! /bin/sh
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
             (system vm trace))

(include-from-path "init_common.scm")
(use-modules (guile-llama-cpp))

(include-from-path "utils.scm")

(define (prompt prompt-text model-path) ;model-path is a gguf file path
  (let ((prompt #f)
        (model-params #f)
        (context-params #f)
        (model #f)
        (ctx #f)
        (n-tokens-max #f)
        (n-ctx #f)
        (tokens #f)
        (tokens-as-byte-vector #f)
        (zero-token-vector #f)
        (n-tokens #f)
        (n-kv-req #f)
        (n-predict 512)  ; total length of the sequence including the prompt
        (batch #f)
        (r #f)
        (result "")
        (n-cur #f)
        (n-decode #f)
        (break #f))
    (llama-backend-init)
    (set! model-params (llama-model-default-params))
    (set! model (llama-load-model-from-file model-path model-params))
    (set! context-params (llama-context-default-params))
    (llama-context-params-seed-set context-params 1234)
    (llama-context-params-n-ctx-set context-params 2048)
    (llama-context-params-n-threads-set context-params 8)
    (llama-context-params-n-threads-batch-set context-params 8)

    (set! ctx (llama-new-context-with-model model context-params))
    (set! n-tokens-max (llama-context-params-n-ctx-get context-params))
    (set! tokens (new-llama-token-array n-tokens-max))
    (set! n-tokens (llama-tokenize (llama-get-model ctx) prompt-text (llama-token-array-cast tokens) n-tokens-max  #t #f))

    (set! n-ctx (llama-n-ctx ctx))
    (set! n-kv-req n-predict)

    (if (> n-kv-req n-ctx)
        ((format #t "error: n_kv_req > n_ctx, the required KV cache size is not big enough\n"))
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
              ((format #t "llama-decode failed"))

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
					;(llama-token-data-array-size-set candidates n-vocab)
                    (do ((token-id 0 (+ token-id 1)))
                        ((>= token-id n-vocab) #t)
                      (let ((token-data (new-llama-token-data))
                            (logits-a (float-array-frompointer logits)))
                        (llama-token-data-id-set token-data token-id)
                        (llama-token-data-logit-set token-data (float-array-getitem logits-a token-id))
                        (llama-token-data-p-set token-data 0.0)

                        (llama-token-data-vector-push! candidates token-data)))
                    (set!  candidates-p (new-llama-token-data-array))
                    (llama-token-data-array-data-set candidates-p (llama-token-data-vector-data candidates))
                    (llama-token-data-array-size-set candidates-p n-vocab)
                    (llama-token-data-array-sorted-set candidates-p #f)
                    (set! new-token-id (llama-sample-token-greedy ctx candidates-p))
                    (set! eog (llama-token-is-eog model new-token-id))
                    (if (or eog (eq? n-cur n-predict))
                        (begin
                          (format #t "~%")  ; end of generation-
                          (set! break #t))  ;need to break out of loop
                        (begin
                          (let ((gen (llama-token-to-piece-return-string ctx new-token-id)))
                            (set! result (string-append result gen ))
                            (format #t "~:a" gen)
                            )
                          (llama-batch-clear batch)
                          (llama-batch-add batch new-token-id n-cur zero-token-vector #t)
                          (set! n-decode (+ n-decode 1))

                          (set! n-cur (+ n-cur 1))
                          (set! r (llama-decode ctx batch))

                          ))
                    ))  ;end main loop
                ))

          (llama-batch-free batch)))

    (delete-llama-token-array tokens)
    (llama-free ctx)
    (llama-free-model model)
    (llama-backend-free)
    result
    ))

(define (main args)
  (let* ((reply #f)
         (prompt-text (cadr args))
         (model-path (caddr args))
	 )
    (format #t "args:  '~:a' ~:a ~%"  prompt-text model-path)
    (set! reply (prompt prompt-text model-path))
    (newline)
    (format #t "LLM reply: ~:a  ~%"  reply)

    (exit 0)))
