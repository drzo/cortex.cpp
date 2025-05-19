
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


(use-modules (ice-9 format)
             ((rnrs) :version (6) #:prefix r6rs:)
             (system vm trace)
             )
(let ((prog_name "guile_llama_cpp")
      (version_string "0.2"))
  (format #t "~:a ~:a Copyright 2024 Li-Cheng (Andy) Tai ~%" prog_name  version_string)
  (format #t "License: GNU LGPL3+ ~%"))

(load-extension "libguile_llama_cpp.so" "scm_init_guile_llama_cpp_module")
(format #t "libguile_llama_cpp.so loaded ~%")

