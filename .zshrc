autoload -Uz compinit
compinit

alias lsl='ls -lh'
alias lsa='ls -alh'

alias gitgraph="git log --graph --full-history --all --color --pretty=format:'%x1b[31m%h%x09%x1b[32m%d%x1b[0m%x20%s - %an %ar'"

parse_git_branch() {
    git branch 2> /dev/null | sed -n -e 's/^\* \(.*\)/[\1]/p'
}
COLOR_DEF='%f'
COLOR_USR='%F{243}'
COLOR_DIR='%F{197}'
COLOR_GIT='%F{39}'
NEWLINE=$'\n'
setopt PROMPT_SUBST
export PROMPT='${COLOR_USR}${COLOR_DIR}%d ${COLOR_DEF}${NEWLINE}${COLOR_GIT}$(parse_git_branch)${COLOR_DEF}%% '

