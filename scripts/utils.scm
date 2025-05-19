

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

;; file meant to be included from other scm files; not standard along script file

(use-modules (ice-9 iconv))     ; for bytevector to string

(define* (llama-token-to-piece-return-string ctx token #:optional (special #t))
  (let* ((check #f)
         (tbv (make-bytevector 8 0))
         (n-tokens (llama-token-to-piece (llama-get-model ctx) token
					 tbv special))
         (results #f))
    (if (< n-tokens 0)
        (begin
          (set! n-tokens (- n-tokens))
          (set! results (make-bytevector  n-tokens))
          (set! check (llama-token-to-piece (llama-get-model ctx) token
					    results special)))  ;TO DO: assert check is n-tokens
        (begin
          (set! results (make-bytevector n-tokens))
          (bytevector-copy! tbv 0 results 0 n-tokens)))

    (utf8->string results)))

(define (llama-batch-clear batch)
  (llama-batch-n-tokens-set batch 0))
