# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/HOME/hku2021_fos4/hku2021_fos4xy_2/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/HOME/hku2021_fos4/hku2021_fos4xy_2/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/HOME/hku2021_fos4/hku2021_fos4xy_2/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/HOME/hku2021_fos4/hku2021_fos4xy_2/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


# >>> mamba initialize >>>
# !! Contents within this block are managed by 'mamba shell init' !!
export MAMBA_EXE='/XYFS01/HOME/hku2021_fos4/hku2021_fos4xy_2/miniforge3/bin/mamba';
export MAMBA_ROOT_PREFIX='/XYFS01/HOME/hku2021_fos4/hku2021_fos4xy_2/miniforge3';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias mamba="$MAMBA_EXE"  # Fallback on help from mamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<
