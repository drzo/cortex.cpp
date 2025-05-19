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
             (ice-9 getopt-long) ; getopt
             (ice-9 threads)  ; for parallel processing
             (ice-9 match)    ; pattern matching
             (ice-9 format)   ; advanced formatting
             (srfi srfi-1)    ; list library
             (srfi srfi-19)   ; date/time library
             (ice-9 pretty-print)) ; for beautiful printing

(include-from-path "init_common.scm")
(use-modules (guile-llama-cpp))

(include-from-path "utils.scm")

;;; ANSI Color codes for beautiful terminal output
(define *color-reset*   "\033[0m")
(define *color-bold*    "\033[1m")
(define *color-dim*     "\033[2m")
(define *color-italic*  "\033[3m") 
(define *color-red*     "\033[31m")
(define *color-green*   "\033[32m")
(define *color-yellow*  "\033[33m")
(define *color-blue*    "\033[34m")
(define *color-magenta* "\033[35m")
(define *color-cyan*    "\033[36m")
(define *color-white*   "\033[37m")

;;; Conversation history stored as an alist with timestamp, role, and content
(define *conversation-history* '())

;;; ASCII Art Banner for an amazing visual experience
(define (display-banner)
  (let ((banner (string-append
                 *color-cyan* "\n"
                 "   ██████╗ ██████╗ ██████╗ ████████╗███████╗██╗  ██╗            \n"
                 "  ██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝            \n"
                 "  ██║     ██║   ██║██████╔╝   ██║   █████╗   ╚███╔╝             \n"
                 "  ██║     ██║   ██║██╔══██╗   ██║   ██╔══╝   ██╔██╗             \n"
                 "  ╚██████╗╚██████╔╝██║  ██║   ██║   ███████╗██╔╝ ██╗            \n"
                 "   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝            \n" 
                 *color-magenta*
                 " ██████╗██╗  ██╗ █████╗ ████████╗    ███╗   ███╗ █████╗  ██████╗ ██╗ ██████╗\n"
                 "██╔════╝██║  ██║██╔══██╗╚══██╔══╝    ████╗ ████║██╔══██╗██╔════╝ ██║██╔════╝\n"
                 "██║     ███████║███████║   ██║       ██╔████╔██║███████║██║  ███╗██║██║     \n"
                 "██║     ██╔══██║██╔══██║   ██║       ██║╚██╔╝██║██╔══██║██║   ██║██║██║     \n"
                 "╚██████╗██║  ██║██║  ██║   ██║       ██║ ╚═╝ ██║██║  ██║╚██████╔╝██║╚██████╗\n"
                 " ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝       ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝ ╚═════╝\n"
                 *color-reset*)))
    (display banner)))

;;; Save conversation history to a file
(define* (save-conversation-history #:optional (filename #f))
  (let* ((timestamp (date->string (current-date) "~Y-~m-~d-~H~M~S"))
         (default-filename (string-append "cortex-chat-" timestamp ".scm"))
         (output-file (or filename default-filename)))
    (with-output-to-file output-file
      (lambda ()
        (pretty-print *conversation-history*)))
    output-file))

;;; Load conversation history from a file
(define (load-conversation-history filename)
  (with-input-from-file filename
    (lambda ()
      (set! *conversation-history* (read)))))

;;; Add entry to conversation history
(define (add-to-history role content)
  (let ((timestamp (current-time)))
    (set! *conversation-history* 
          (append *conversation-history* 
                  (list (list (cons 'timestamp timestamp)
                              (cons 'role role)
                              (cons 'content content)))))))

;;; Format history into a prompt for context
(define (format-history-as-prompt)
  (string-join
   (map (lambda (entry)
          (let ((role (assoc-ref entry 'role))
                (content (assoc-ref entry 'content)))
            (string-append 
             (if (eq? role 'system) "System: " 
                 (if (eq? role 'user) "User: " "Assistant: "))
             content)))
        *conversation-history*)
   "\n\n"))

;;; Animate text to simulate typing effect
(define (animate-text text)
  (let ((len (string-length text)))
    (do ((i 0 (+ i 1)))
        ((>= i len))
      (display (string-ref text i))
      (force-output)
      (usleep (+ 10000 (random 40000))))))