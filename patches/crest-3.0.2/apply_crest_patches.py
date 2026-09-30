"""把 P1/P2/P3 施加到 CREST 的**构建副本**上。

vendored 源码 (`../source-code/crest-master`) 一个字都不改 —— 这是这套补丁的约束。
本脚本只动 ~/crest_build/crest-patched。
"""
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

# ---------------- P1 + P2: src/iomod.F90 ----------------
p = root / "src/iomod.F90"
s = p.read_text()

OLD_MAKEDIR = """  function makedir(str)
    implicit none
    integer :: makedir
    character(len=*) :: str
    makedir = mkdir(str//char(0),int(o'770',c_int16_t)) !create new directory
    return
  end function makedir"""

NEW_MAKEDIR = """  function makedir(str)
!>--- P1 (stage 0 patch): create the directory RECURSIVELY and return a
!>    status that callers can actually check.
!>    Upstream called the C mkdir() exactly once, non-recursively, and every
!>    call site threw the return value away.  A nested calculation space such
!>    as calculation.level.1 followed by _1 therefore failed silently with
!>    ENOENT, and the very next statement (remove) killed the program instead
!>    of reporting the problem.
!>    Returns 0 on success or if the directory already exists, non-zero else.
    implicit none
    integer :: makedir
    character(len=*) :: str
    integer :: i,l
    character(len=:),allocatable :: path
    makedir = 0
    l = len_trim(str)
    if (l < 1) return
    path = trim(str)
    !>--- walk the path and create every missing component in turn
    do i = 2,l
      if (path(i:i) == '/') then
        if (.not.directory_exist(path(1:i-1))) then
          makedir = mkdir(path(1:i-1)//char(0),int(o'770',c_int16_t))
          if (makedir /= 0 .and. .not.directory_exist(path(1:i-1))) return
        end if
      end if
    end do
    if (directory_exist(path)) then
      makedir = 0
      return
    end if
    makedir = mkdir(path//char(0),int(o'770',c_int16_t))
    !>--- losing a race with another thread is not an error if it now exists
    if (makedir /= 0 .and. directory_exist(path)) makedir = 0
    return
  end function makedir"""

OLD_REMOVE = """  subroutine remove(fname)
    implicit none
    character(len=*) :: fname
    integer :: ich,och
    logical :: ex
    if (index(fname,'*') .eq. 0) then
      open (newunit=ich,file=fname)
      close (ich,status='delete')
    end if
  end subroutine remove"""

NEW_REMOVE = """  subroutine remove(fname)
!>--- P2 (stage 0 patch): deleting something that is not there must be
!>    a no-op, not a crash.  Upstream opened the file with no iostat, so a
!>    missing PARENT DIRECTORY produced
!>      Fortran runtime error: Cannot open file ...
!>    and killed the run.  Ask first, and keep iostat as a second guard.
    implicit none
    character(len=*) :: fname
    integer :: ich,io
    logical :: ex
    if (index(fname,'*') .eq. 0) then
      inquire (file=fname,exist=ex)
      if (.not.ex) return
      open (newunit=ich,file=fname,iostat=io)
      if (io /= 0) return
      close (ich,status='delete',iostat=io)
    end if
  end subroutine remove"""

assert OLD_MAKEDIR in s, "P1 锚点没找到"
assert OLD_REMOVE in s, "P2 锚点没找到"
s = s.replace(OLD_MAKEDIR, NEW_MAKEDIR).replace(OLD_REMOVE, NEW_REMOVE)
p.write_text(s)
print("P1 + P2 -> src/iomod.F90")

# ---------------- P3: 六处格式串缺逗号 ----------------
BAD = "'(1x,\"(\"f7.2\"%)\")'"
GOOD = "'(1x,\"(\",f7.2,\"%)\")'"
total = 0
for f in ("src/optimize/ancopt.f90", "src/optimize/rfo.f90"):
    q = root / f
    t = q.read_text()
    k = t.count(BAD)
    q.write_text(t.replace(BAD, GOOD))
    total += k
    print("P3 -> {}: {} 处".format(f, k))
assert total == 6, "P3 应当命中 6 处, 实际 {}".format(total)
print("P3 合计 {} 处".format(total))
