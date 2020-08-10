!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!*********************************************************************!
!*                                                                  **!
!*                          connections                             **!
!*                      ====================                        **!
!*                                                                  **!
!*  Ricardo Mendes Ribeiro                                          **!
!*  Date: Aug: 2020                                                 **!
!*  Description: Main program that compares the wavefunctions       **!
!*             of a set and make connections                        **!
!*                                                                  **!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!************************************************************************
! compiles with command
! py -m numpy.f2py -c connections.f90 -m connections
!
SUBROUTINE connect(nk, nbnd, nr, npr, neighbor, wfcdirectory, phase)

IMPLICIT NONE

  INTEGER(KIND=4) :: i, banda, banda1
  INTEGER(KIND=4), INTENT(IN) :: nk, nr
  INTEGER :: IOstatus
  INTEGER, INTENT(IN) :: nbnd, npr, neighbor
  CHARACTER(LEN=20) :: fmt1, fmt6, fmt7
  CHARACTER(LEN=50), INTENT(IN) :: wfcdirectory
  CHARACTER(LEN=50) :: wfcdirector
  CHARACTER(LEN=15) :: str1, str2
  CHARACTER(LEN=50) :: infile
  REAL(KIND=8),ALLOCATABLE :: psir(:,:,:), psii(:,:,:)
  REAL(KIND=8),ALLOCATABLE :: dp(:,:)
  COMPLEX(KIND=8), DIMENSION(0:nr-1,0:1), INTENT(IN) :: phase
  COMPLEX(KIND=8),ALLOCATABLE :: dphase(:),dpc(:,:)
  REAL(KIND=8) :: tol0, tol1

  ! nk - number of the reference k-point
  ! neighbor - number of the neighbor to compare with
  ! nbnd - total number of bands
  ! nr - number of points in r-space
  ! phase - array with e^{-ik.r}
  ! wfcdirectory - directory where wfc files are
  ! npr - number of processors for parallel runs

  fmt1 = '(2f22.16)'
  fmt6 = '(4i6,1f15.5)'
  fmt7 = '(4i6,2f25.10)'

  wfcdirector = trim(wfcdirectory)
  tol0 = 0.9
  tol1 = 0.85

  ALLOCATE(dphase(0:nr-1))
  dphase = (0,0)

!  WRITE(*,*)nk,neighbor,nbnd,nr,npr,wfcdirectory


  ALLOCATE(dp(1:nbnd,1:nbnd),dpc(1:nbnd,1:nbnd))
  dp = 0                  ! modulus of dot product of wfc
  dpc = (0,0)             ! dot product of wfc (complex)


! ****************************************************************************
  
!  WRITE(*,*)' Start reading files'

  ALLOCATE(psir(0:1,1:nbnd,0:nr-1), psii(0:1,1:nbnd,0:nr-1))

  WRITE(str1,*) nk
  DO banda = 1,nbnd
    WRITE(str2,*) banda
    infile = trim(wfcdirector)//'k0'//trim(adjustl(str1))//'b0'//trim(adjustl(str2))//'.wfc'
!    WRITE(*,*)banda,infile
    OPEN(FILE=infile,UNIT=5,STATUS='OLD')
    DO i = 0,nr-1
      READ(UNIT=5,FMT=fmt1,IOSTAT=IOstatus) psir(0,banda,i),psii(0,banda,i)
    ENDDO
    CLOSE(UNIT=5)
  ENDDO

  WRITE(str1,*) neighbor
  DO banda = 1,nbnd
    WRITE(str2,*) banda
    infile = trim(wfcdirector)//'k0'//trim(adjustl(str1))//'b0'//trim(adjustl(str2))//'.wfc'
!    WRITE(*,*)banda,infile
    OPEN(FILE=infile,UNIT=5,STATUS='OLD')
    DO i = 0,nr-1
      READ(UNIT=5,FMT=fmt1,IOSTAT=IOstatus) psir(1,banda,i),psii(1,banda,i)
    ENDDO
    CLOSE(UNIT=5)
  ENDDO

!  WRITE(*,*)' Finished reading files'

! ****************************************************************************
!  WRITE(*,*)' Start calculating connections'
  dphase(:) = phase(:,0)*CONJG(phase(:,1))
  DO banda = 1,nbnd
    DO banda1 = 1,nbnd
      dpc(banda, banda1) = SUM(dphase(:)* &
                           CMPLX(psir(0,banda,:),psii(0,banda,:),KIND=8)*  &
                           CMPLX(psir(1,banda1,:),-psii(1,banda1,:),KIND=8))
      dp(banda, banda1) = ABS(dpc(banda, banda1))
!      write(*,*)nk,banda,banda1,dp(banda, banda1) 
    ENDDO
  ENDDO
  dpc = dpc/FLOAT(nr)
  dp = dp/FLOAT(nr)

  OPEN(UNIT=9,FILE='dp.dat',STATUS='OLD',ACCESS='APPEND')
  DO banda = 1,nbnd
    DO banda1 = 1,nbnd
      WRITE(9,fmt6) nk,neighbor,banda,banda1,dp(banda, banda1)
    ENDDO
  ENDDO
  CLOSE(UNIT=9)
  OPEN(UNIT=9,FILE='dpc.dat',STATUS='OLD',ACCESS='APPEND')
  DO banda = 1,nbnd
    DO banda1 = 1,nbnd
      WRITE(9,fmt7) nk,neighbor,banda,banda1,dpc(banda, banda1)
    ENDDO
  ENDDO
  CLOSE(UNIT=9)



! ****************************************************************************


END SUBROUTINE connect

