#!/bin/bash -e
gsed "s/^M//g" \
| tr '\n' ' ' \
| gsed "s/\t/ /g" \
| gsed "s/(\([^()]*\))/\n\1\n/g" \
| gsed "s/\"\([^\"]*\)\"/\n\1\n/g" \
| gsed "s/\"\([^\']*\)\'/\n\1\n/g" \
| gsed "s/\[[^][]*\]//g" \
| gsed "s/\[,:\/\` ]/ /g" \
| gsed "s/[\?\!\.;]/\n/g" \
| gsed "s/[^a-zA-Z0-9 ]/ /g" \
| gsed "s/./\L&/g" \
| gsed "s/ [  ]*/ /g" \
| gsed "s/^[ \t]*/ /g" \
| gsed "s/[\t ]*$//g" \
| gsed "/^$/d" \
| gsed "/^[^ ]*$/d" \
| gsed "s/[^[:print:]]//g" \
| gsed "s/^/<s> /" \
| gsed "s/$/ <\/s>/" \
