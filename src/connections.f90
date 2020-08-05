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

PROGRAM connect

IMPLICIT NONE

  INTEGER(KIND=4) :: i,j,l, nk, banda, banda1
  INTEGER :: IOstatus,n0(10),n1(10),n2(10),n3(10)
  INTEGER :: nbnd, nr, npr, neighbor
  CHARACTER(LEN=20) :: fmt1, fmt2, fmt3, fmt4, fmt5, fmt6, fmt7
  CHARACTER(LEN=50) :: dummy, wfcdirectory
  CHARACTER(LEN=15) :: str1, str2
  CHARACTER(LEN=50) :: infile
  REAL(KIND=8),ALLOCATABLE :: psir(:,:,:), psii(:,:,:)
  REAL(KIND=8),ALLOCATABLE :: dp(:,:)
  COMPLEX(KIND=8),ALLOCATABLE :: phase(:,:), dphase(:),dpc(:,:)
  INTEGER(KIND=4),ALLOCATABLE :: connections(:,:,:), connections1(:,:,:), connections2(:,:,:)
  REAL(KIND=8) :: tol0, tol1, tol2
  COMPLEX(KIND=8),PARAMETER :: pi2=(0,-6.28318530717958647688)

  NAMELIST / input / nk, nbnd, nr, npr, neighbor, wfcdirectory, phase


  ! nk - number of the reference k-point
  ! neighbor - number of the neighbor to compare with
  ! nbnd - total number of bands
  ! nr - number of points in r-space
  ! phase - array with e^{-ik.r}
  ! wfcdirectory - directory where wfc files are
  ! npr - number of processors for parallel runs

  fmt1 = '(2f22.16)'
  fmt2 = '(i4)'
  fmt3 = '(3i6)'
  fmt4 = '(5i6)'
  fmt5 = '(6i6)'
  fmt6 = '(4i6,1f14.5)'
  fmt7 = '(4i6,2f14.5)'

  wfcdirectory = trim(wfcdirectory)
  tol0 = 0.9
  tol1 = 0.85
  tol2 = 0.8

  OPEN(FILE='tmp',UNIT=2,STATUS='OLD')
  READ(2,*) nr
  CLOSE(UNIT=2)
  ALLOCATE(phase(0:nr,0:1),dphase(0:nr))
  phase = (0,0)

  READ(*,NML=input)

!  WRITE(*,*)nk,nbnd,nr,npr,neighbor,wfcdirectory


  ALLOCATE(dp(1:nbnd,1:nbnd),dpc(1:nbnd,1:nbnd))
  dp = 0                  ! modulus of dot product of wfc
  dpc = (0,0)             ! dot product of wfc (complex)


! ALLOCATE(connections(0:nks-1,1:nbnd,0:3),connections2(0:nks-1,1:nbnd,0:3))
! ALLOCATE(connections1(0:nks-1,1:nbnd,0:3))
! connections = 0
! connections1 = 0
! connections2 = 0


! ****************************************************************************
  
!  WRITE(*,*)' Start reading files'

  ALLOCATE(psir(0:1,1:nbnd,1:nr), psii(0:1,1:nbnd,1:nr))

  WRITE(str1,*) nk
  DO banda = 1,nbnd
    WRITE(str2,*) banda
    infile = trim(wfcdirectory)//'k0'//trim(adjustl(str1))//'b0'//trim(adjustl(str2))//'.wfc'
!    WRITE(*,*)banda,infile
    OPEN(FILE=infile,UNIT=5,STATUS='OLD')
    DO i = 1,nr
      READ(UNIT=5,FMT=fmt1,IOSTAT=IOstatus) psir(0,banda,i),psii(0,banda,i)
    ENDDO
    CLOSE(UNIT=5)
  ENDDO

  WRITE(str1,*) neighbor
  DO banda = 1,nbnd
    WRITE(str2,*) banda
    infile = trim(wfcdirectory)//'k0'//trim(adjustl(str1))//'b0'//trim(adjustl(str2))//'.wfc'
!    WRITE(*,*)banda,infile
    OPEN(FILE=infile,UNIT=5,STATUS='OLD')
    DO i = 1,nr
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


! DO nk1 = 0,nks-1
!   DO banda = 1,nbnd
!     DO banda1 = 1,nbnd
!       DO i = 0,3
!         IF (connections(nk1,banda,i) == 0 .AND. dp(nk1,banda, banda1,i) > tol1) THEN
!           connections1(nk1,banda,i) = banda1
!         ENDIF
!       ENDDO
!     ENDDO
!   ENDDO
! ENDDO

! DO nk1 = 0,nks-1
!   DO banda = 1,nbnd
!     DO banda1 = 1,nbnd
!       DO i = 0,3
!         IF (connections(nk1,banda,i) == 0 .AND. connections1(nk1,banda,i) == 0 &
!              .AND. dp(nk1,banda, banda1,i) > tol2) THEN
!           connections2(nk1,banda,i) = banda1
!         ENDIF
!       ENDDO
!     ENDDO
!   ENDDO
! ENDDO

! ****************************************************************************
! Save calculated connections to file
! OPEN(UNIT=9,FILE='connections',STATUS='UNKNOWN')
! DO nk1 = 0,nks-1
!   DO banda = 1,nbnd
!     WRITE(9,fmt5)nk1,banda,connections(nk1,banda,:)
!   ENDDO
! ENDDO    
! CLOSE(UNIT=9)
! OPEN(UNIT=9,FILE='connections1',STATUS='UNKNOWN')
! DO nk1 = 0,nks-1
!   DO banda = 1,nbnd
!     WRITE(9,fmt5)nk1,banda,connections1(nk1,banda,:)
!   ENDDO
! ENDDO    
! CLOSE(UNIT=9)
! OPEN(UNIT=9,FILE='connections2',STATUS='UNKNOWN')
! DO nk1 = 0,nks-1
!   DO banda = 1,nbnd
!     WRITE(9,fmt5)nk1,banda,connections2(nk1,banda,:)
!   ENDDO
! ENDDO    
! CLOSE(UNIT=9)
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


END PROGRAM connect

