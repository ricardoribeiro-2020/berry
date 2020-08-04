!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!*********************************************************************!
!*                                                                  **!
!*                          connections                             **!
!*                      ====================                        **!
!*                                                                  **!
!*  Ricardo Mendes Ribeiro                                          **!
!*  Date: Jan, 2020                                                 **!
!*  Description: Main program that compares the wavefunctions       **!
!*             of a set and make connections                        **!
!*                                                                  **!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!************************************************************************

PROGRAM connect

IMPLICIT NONE

  INTEGER(KIND=4) :: i,j,l, nk, nk1, banda, banda1
  INTEGER :: IOstatus,n0(10),n1(10),n2(10),n3(10)
  INTEGER :: nbnd, nks, nr, np, neighbor
  CHARACTER(LEN=20) :: fmt1, fmt2, fmt3, fmt4, fmt5, fmt6, fmt7
  CHARACTER(LEN=50) :: dummy, wfcdirectory
  CHARACTER(LEN=15) :: str1, str2
  CHARACTER(LEN=50) :: infile
  REAL(KIND=8),ALLOCATABLE :: psir(:,:,:), psii(:,:,:)
  REAL(KIND=8),ALLOCATABLE :: dp(:,:,:,:)
  COMPLEX(KIND=8),ALLOCATABLE :: phase(:), dphase(:),dpc(:,:,:,:)
  INTEGER(KIND=4),ALLOCATABLE :: connections(:,:,:), connections1(:,:,:), connections2(:,:,:)
  REAL(KIND=8) :: tol0, tol1, tol2
  COMPLEX(KIND=8),PARAMETER :: pi2=(0,-6.28318530717958647688)

  NAMELIST / input / nk, nbnd, np, nr, neighbor, wfcdirectory, phase

  fmt1 = '(3f14.8,3f22.16)'
  fmt2 = '(i4)'
  fmt3 = '(3i6)'
  fmt4 = '(5i6)'
  fmt5 = '(6i6)'
  fmt6 = '(3i6,4f14.5)'
  fmt7 = '(3i6,8f14.5)'

  wfcdirectory = 'wfc'
  tol0 = 0.9
  tol1 = 0.85
  tol2 = 0.8

  OPEN(FILE='tmp',UNIT=2,STATUS='OLD')
  READ(2,*) nr
  CLOSE(UNIT=2)
  ALLOCATE(phase(0:nr))

  READ(*,NML=input)

!  write(*,*)nk,nbnd,np,nr,wfcdirectory,neighbor


stop

  ALLOCATE(dp(0:nks-1,1:nbnd,1:nbnd,0:3),dpc(0:nks-1,1:nbnd,1:nbnd,0:3))
  ALLOCATE(connections(0:nks-1,1:nbnd,0:3),connections2(0:nks-1,1:nbnd,0:3))
  ALLOCATE(connections1(0:nks-1,1:nbnd,0:3))

  dpc = (0,0)
  connections = 0
  connections1 = 0
  connections2 = 0


! ****************************************************************************
  nr = nr + 1
  WRITE(*,*)' Start reading files'
  ALLOCATE(psir(0:nks-1,1:nbnd,1:nr), psii(0:nks-1,1:nbnd,1:nr))
  DO nk1 = 0,nks-1
    WRITE(*,*)' Reading files of k-point ',nk1
    DO banda = 1,nbnd
      WRITE(str1,*) nk1
      WRITE(str2,*) banda
!      infile(nk1,banda) = trim(wfcdirectory)//'/k000'//trim(adjustl(str1))//'b000'//trim(adjustl(str2))//'.wfc'
!      WRITE(*,*)nk1,banda,infile(nk1,banda)
      OPEN(FILE=infile(nk1,banda),UNIT=5,STATUS='OLD')
      i = 1
      IOstatus = 0
      DO WHILE (IOstatus == 0)
        READ(UNIT=5,FMT=fmt1,IOSTAT=IOstatus) psir(nk1,banda,i),psii(nk1,banda,i)
        i = i + 1
      ENDDO
      CLOSE(UNIT=5)
    ENDDO
  ENDDO
  WRITE(*,*)' Finished reading files'

! ****************************************************************************
  WRITE(*,*)' Start calculating connections'
  DO nk1 = 0,nks-1
    WRITE(*,*)' K-point ',nk1
    IF (n0(nk1) == -1) THEN
      dp(nk1,:,:,0) = 0.0
!     rho(nk1,:,:,0) = 9E10
      connections(nk1,:,0) = -1
    ELSE
!      dphase(:) = phase(:,nk1)*CONJG(phase(:,n0(nk1)))
      DO banda = 1,nbnd
        DO banda1 = 1,nbnd
          dpc(nk1,banda, banda1,0) = SUM(dphase(:)* &
                                    CMPLX(psir(nk1,banda,:),psii(nk1,banda,:),KIND=8)*  &
                                    CMPLX(psir(n0(nk1),banda1,:),-psii(n0(nk1),banda1,:),KIND=8))
          dp(nk1,banda, banda1,0) = ABS(dpc(nk1,banda, banda1,0))
          IF (dp(nk1,banda, banda1,0) > tol0) THEN
            connections(nk1,banda,0) = banda1
!            write(*,*)nk1,banda,banda1,dp(nk1,banda, banda1,0) 
          ENDIF
        ENDDO
      ENDDO
    ENDIF

    IF (n1(nk1) == -1) THEN
      dp(nk1,:,:,1) = 0.0
!     rho(nk1,:,:,1) = 9E10
      connections(nk1,:,1) = -1
    ELSE
!      dphase(:) = phase(:,nk1)*CONJG(phase(:,n1(nk1)))
      DO banda = 1,nbnd
        DO banda1 = 1,nbnd
          dpc(nk1,banda, banda1,1) = SUM(dphase(:)* &
                                    CMPLX(psir(nk1,banda,:),psii(nk1,banda,:),KIND=8)*  &
                                    CMPLX(psir(n1(nk1),banda1,:),-psii(n1(nk1),banda1,:),KIND=8))
          dp(nk1,banda, banda1,1) = ABS(dpc(nk1,banda, banda1,1))
          IF (dp(nk1,banda, banda1,1) > tol0) THEN
            connections(nk1,banda,1) = banda1
!            write(*,*)nk1,banda,banda1,dp(nk1,banda, banda1,1)
          ENDIF
        ENDDO
      ENDDO
    ENDIF

    IF (n2(nk1) == -1) THEN
      dp(nk1,:,:,2) = 0.0
!     rho(nk1,:,:,2) = 9E10
      connections(nk1,:,2) = -1
    ELSE
!      dphase(:) = phase(:,nk1)*CONJG(phase(:,n2(nk1)))
      DO banda = 1,nbnd
        DO banda1 = 1,nbnd
          dpc(nk1,banda, banda1,2) = SUM(dphase(:)* &
                                    CMPLX(psir(nk1,banda,:),psii(nk1,banda,:),KIND=8)*  &
                                    CMPLX(psir(n2(nk1),banda1,:),-psii(n2(nk1),banda1,:),KIND=8))
          dp(nk1,banda, banda1,2) = ABS(dpc(nk1,banda, banda1,2))
          IF (dp(nk1,banda, banda1,2) > tol0) THEN
            connections(nk1,banda,2) = banda1
!            write(*,*)nk1,banda,banda1,dp(nk1,banda, banda1,2)
          ENDIF
        ENDDO
      ENDDO
    ENDIF

    IF (n3(nk1) == -1) THEN
      dp(nk1,:,:,3) = 0.0
!     rho(nk1,:,:,3) = 9E10
      connections(nk1,:,3) = -1
    ELSE
!      dphase(:) = phase(:,nk1)*CONJG(phase(:,n3(nk1)))
      DO banda = 1,nbnd
        DO banda1 = 1,nbnd
          dpc(nk1,banda, banda1,3) = SUM(dphase(:)* &
                                    CMPLX(psir(nk1,banda,:),psii(nk1,banda,:),KIND=8)*  &
                                    CMPLX(psir(n3(nk1),banda1,:),-psii(n3(nk1),banda1,:),KIND=8))
          dp(nk1,banda, banda1,3) = ABS(dpc(nk1,banda, banda1,3))
          IF (dp(nk1,banda, banda1,3) > tol0) THEN
            connections(nk1,banda,3) = banda1
!            write(*,*)nk1,banda,banda1,dp(nk1,banda, banda1,3)
          ENDIF
        ENDDO
      ENDDO
    ENDIF
  ENDDO

  DO nk1 = 0,nks-1
    DO banda = 1,nbnd
      DO banda1 = 1,nbnd
        DO i = 0,3
          IF (connections(nk1,banda,i) == 0 .AND. dp(nk1,banda, banda1,i) > tol1) THEN
            connections1(nk1,banda,i) = banda1
          ENDIF
        ENDDO
      ENDDO
    ENDDO
  ENDDO

  DO nk1 = 0,nks-1
    DO banda = 1,nbnd
      DO banda1 = 1,nbnd
        DO i = 0,3
          IF (connections(nk1,banda,i) == 0 .AND. connections1(nk1,banda,i) == 0 &
               .AND. dp(nk1,banda, banda1,i) > tol2) THEN
            connections2(nk1,banda,i) = banda1
          ENDIF
        ENDDO
      ENDDO
    ENDDO
  ENDDO

! ****************************************************************************
! Save calculated connections to file
  OPEN(UNIT=9,FILE='connections',STATUS='UNKNOWN')
  DO nk1 = 0,nks-1
    DO banda = 1,nbnd
      WRITE(9,fmt5)nk1,banda,connections(nk1,banda,:)
    ENDDO
  ENDDO    
  CLOSE(UNIT=9)
  OPEN(UNIT=9,FILE='connections1',STATUS='UNKNOWN')
  DO nk1 = 0,nks-1
    DO banda = 1,nbnd
      WRITE(9,fmt5)nk1,banda,connections1(nk1,banda,:)
    ENDDO
  ENDDO    
  CLOSE(UNIT=9)
  OPEN(UNIT=9,FILE='connections2',STATUS='UNKNOWN')
  DO nk1 = 0,nks-1
    DO banda = 1,nbnd
      WRITE(9,fmt5)nk1,banda,connections2(nk1,banda,:)
    ENDDO
  ENDDO    
  CLOSE(UNIT=9)
  OPEN(UNIT=9,FILE='dp.dat',STATUS='UNKNOWN')
  DO nk1 = 0,nks-1
    DO banda = 1,nbnd
      DO banda1 = 1,nbnd
        WRITE(9,fmt6) nk1,banda,banda1,dp(nk1,banda, banda1,:)
      ENDDO
    ENDDO
  ENDDO
  CLOSE(UNIT=9)
  OPEN(UNIT=9,FILE='dpc.dat',STATUS='UNKNOWN')
  DO nk1 = 0,nks-1
    DO banda = 1,nbnd
      DO banda1 = 1,nbnd
        WRITE(9,fmt7) nk1,banda,banda1,dpc(nk1,banda, banda1,:)
      ENDDO
    ENDDO
  ENDDO
  CLOSE(UNIT=9)



! ****************************************************************************


END PROGRAM connect

