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
             (system foreign)) ; for FFI

(include-from-path "init_common.scm")
(use-modules (guile-llama-cpp))

(include-from-path "utils.scm")

(define (tokenize prompt-text model-path) ;model-path is a gguf file path
  (let ((prompt #f)
        (model-params #f)
        (context-params #f)
        (model #f)
        (ctx #f)
        (n-tokens-max #f)
        (tokens #f)
        (n-tokens #f))
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
    (format #t "prompt text  '~:a'  prompt length ~:a   n tokens  ~:a ~%"  prompt-text (string-length prompt-text) n-tokens)
    (do ((i 0 (+ i 1)))
        ((>= i  n-tokens) #t)
      (format #t "~:a " (llama-token-array-getitem tokens i))
      (format #t "~:a ~%" (llama-token-to-piece-return-string ctx (llama-token-array-getitem tokens i))))

    (llama-free ctx)
    (llama-free-model model)
    (llama-backend-free)
    n-tokens
    ))
(define (main args)
  (let* ((t #f)
         (prompt-text (cadr args))
         (model-path (caddr args))
	 )
    (format #t "args:  '~:a' ~:a ~%"  prompt-text model-path)
    (set! t (tokenize prompt-text model-path))
    (format #t "tokenize result: ~:a  ~%"  t)

    (exit 0)))
