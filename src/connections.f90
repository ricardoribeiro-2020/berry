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
  INTEGER :: IOstatus
  INTEGER :: numero_kx, numero_ky, numero_kz
  INTEGER :: nbands, nks, nr, nrx, nry, nrz
  INTEGER(KIND=4),ALLOCATABLE :: n0(:),n1(:),n2(:),n3(:)
  CHARACTER(LEN=20) :: fmt1, fmt2, fmt3, fmt4, fmt5, fmt6, fmt7
  CHARACTER(LEN=50) :: dummy, wfcdirectory
  CHARACTER(LEN=15) :: str1, str2
  CHARACTER(LEN=50),ALLOCATABLE :: infile(:,:)
  REAL(KIND=8),ALLOCATABLE :: psi2(:,:,:), psir(:,:,:), psii(:,:,:)
  REAL(KIND=8) :: a,b,c
  REAL(KIND=8),ALLOCATABLE :: dp(:,:,:,:)    !, rho(:,:,:,:)
  REAL(KIND=8),ALLOCATABLE :: kx(:), ky(:), kz(:)
  REAL(KIND=8),ALLOCATABLE :: rx(:), ry(:), rz(:)
  COMPLEX(KIND=8),ALLOCATABLE :: phase(:,:), dphase(:),dpc(:,:,:,:)
  INTEGER(KIND=4),ALLOCATABLE :: connections(:,:,:), connections1(:,:,:), connections2(:,:,:)
  REAL(KIND=8) :: tol0, tol1, tol2
  REAL(KIND=8) :: ax,ay,az,bx,by,bz,cx,cy,cz
  COMPLEX(KIND=8),PARAMETER :: pi2=(0,-6.28318530717958647688)

  fmt1 = '(3f14.8,3f22.16)'
  fmt2 = '(i4)'
  fmt3 = '(3i6)'
  fmt4 = '(5i6)'
  fmt5 = '(6i6)'
  fmt6 = '(3i6,4f14.5)'
  fmt7 = '(3i6,8f14.5)'

  wfcdirectory = 'wfc'
!  nr = 86400+1 !68544 !45504
  tol0 = 0.9
  tol1 = 0.85
  tol2 = 0.8

  OPEN(UNIT=1,FILE='connections.dat',STATUS='UNKNOWN')   ! File for relevant data

  WRITE(*,*) ' Reading from file kindex'
  OPEN(FILE='wfc/kindex', UNIT=2,STATUS="OLD")
  READ(2,*) numero_kx, numero_ky, numero_kz
  WRITE(1,*) numero_kx, numero_ky, numero_kz
  CLOSE(UNIT=2)

  WRITE(*,*) ' Reading number of bands from file wfcdata.dat '
  OPEN(FILE='wfcdata.dat', UNIT=2,STATUS="OLD")
!   Read useless data 
  DO WHILE (dummy .ne. '# name: nbands')
    READ(2,'(A)') dummy
  ENDDO
  READ(2,*) dummy
  READ(2,*) nbands
  WRITE(*,*) ' Number of bands ',nbands
  WRITE(1,*) nbands

  DO WHILE (dummy .ne. '# name: nr1x')
    READ(2,'(A)') dummy
  ENDDO
  READ(2,*) dummy
  READ(2,*) nrx

  DO WHILE (dummy .ne. '# name: nr2x')
    READ(2,'(A)') dummy
  ENDDO
  READ(2,*) dummy
  READ(2,*) nry

  DO WHILE (dummy .ne. '# name: nr3x')
    READ(2,'(A)') dummy
  ENDDO
  READ(2,*) dummy
  READ(2,*) nrz

  nr = nrx*nry*nrz

  WRITE(*,*) ' Number of points in R space: ',nr

  CLOSE(UNIT=2)

! OPEN(FILE='dft/scf.in', UNIT=2, STATUS='OLD')
! DO WHILE (dummy .ne. 'CELL_PARAMETERS alat')
!   READ(2,'(A)') dummy
! ENDDO
! READ(2,*) ax,ay,az
! READ(2,*) bx,by,bz
! READ(2,*) cx,cy,cz
! CLOSE(UNIT=2)

  l=0
  nks = numero_kx*numero_ky*numero_kz
  WRITE(*,*) ' Number of k-points: ', nks
  WRITE(1,*) nks
  WRITE(*,*) ' Size of wfc: ', nr
  WRITE(1,*) nr
  WRITE(*,*)

  CLOSE(UNIT=1)

  ALLOCATE(n0(0:nks),n1(0:nks),n2(0:nks),n3(0:nks))
  ALLOCATE(infile(0:nks-1,1:nbands))
  ALLOCATE(dp(0:nks-1,1:nbands,1:nbands,0:3),dpc(0:nks-1,1:nbands,1:nbands,0:3))!, rho(0:nks-1,1:nbands,1:nbands,0:3))
  ALLOCATE(connections(0:nks-1,1:nbands,0:3),connections2(0:nks-1,1:nbands,0:3))
  ALLOCATE(connections1(0:nks-1,1:nbands,0:3))
  ALLOCATE(phase(0:nr,0:nks-1),dphase(0:nr))
  ALLOCATE(kx(0:nks-1), ky(0:nks-1), kz(0:nks-1))
  ALLOCATE(rx(0:nr), ry(0:nr), rz(0:nr))

  OPEN(FILE='wfc/k_points', UNIT=2,STATUS="OLD")
  DO i = 0,nks-1
    READ(2,*) kx(i), ky(i), kz(i)
  ENDDO
  CLOSE(UNIT=2)

  OPEN(FILE='wfc/rindex', UNIT=2,STATUS="OLD")
  DO i = 0,nr-1
    READ(2,*) j,rx(i), ry(i), rz(i)
  ENDDO
  CLOSE(UNIT=2)

  phase = (0,0)
  DO i = 0,nr-1
    DO j = 0,nks-1
      phase(i,j) = EXP(pi2*(rx(i)*kx(j) + ry(i)*ky(j) + rz(i)*kz(j)))
    ENDDO
  ENDDO

  dpc = (0,0)
  connections = 0
  connections1 = 0
  connections2 = 0

  OPEN(UNIT=3,FILE='neighbors',STATUS='UNKNOWN')
  nk = -1
  DO j = 0,numero_ky-1
    DO i = 0,numero_kx-1
      nk = nk + 1
      IF (i == 0) THEN 
        n0(nk) = -1
      ELSE
        n0(nk) = nk - 1
      ENDIF
      IF (j == 0) THEN
        n1(nk) = -1
      ELSE
        n1(nk) = nk - numero_kx
      ENDIF
      IF (i == numero_kx-1) THEN
        n2(nk) = -1
      ELSE
        n2(nk) = nk + 1
      ENDIF
      IF (j == numero_ky-1) THEN
        n3(nk) = -1
      ELSE
        n3(nk) = nk + numero_kx
      ENDIF
      WRITE(3,fmt4) nk,n0(nk),n1(nk),n2(nk),n3(nk)
    ENDDO
  ENDDO
  CLOSE(UNIT=3)

! ****************************************************************************
  nr = nr + 1
  WRITE(*,*)' Start reading files'
  ALLOCATE(psi2(0:nks-1,1:nbands,1:nr), psir(0:nks-1,1:nbands,1:nr), psii(0:nks-1,1:nbands,1:nr))
  DO nk1 = 0,nks-1
    WRITE(*,*)' Reading files of k-point ',nk1
    DO banda = 1,nbands
      WRITE(str1,*) nk1
      WRITE(str2,*) banda
      infile(nk1,banda) = trim(wfcdirectory)//'/k000'//trim(adjustl(str1))//'b000'//trim(adjustl(str2))//'.wfc'
!      WRITE(*,*)nk1,banda,infile(nk1,banda)
      OPEN(FILE=infile(nk1,banda),UNIT=5,STATUS='OLD')
      i = 1
      IOstatus = 0
      DO WHILE (IOstatus == 0)
        READ(UNIT=5,FMT=fmt1,IOSTAT=IOstatus) a,b,c,psi2(nk1,banda,i),psir(nk1,banda,i),psii(nk1,banda,i)
!        write(*,*)a,b,c,psi2(nk1,banda,i),psir(nk1,banda,i),psii(nk1,banda,i)
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
      dphase(:) = phase(:,nk1)*CONJG(phase(:,n0(nk1)))
      DO banda = 1,nbands
        DO banda1 = 1,nbands
!         rho(nk1,banda, banda1,0) = SUM((psi2(nk1,banda,:) - psi2(n0(nk1),banda1,:))**2)
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
      dphase(:) = phase(:,nk1)*CONJG(phase(:,n1(nk1)))
      DO banda = 1,nbands
        DO banda1 = 1,nbands
!         rho(nk1,banda, banda1,1) = SUM((psi2(nk1,banda,:) - psi2(n1(nk1),banda1,:))**2)
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
      dphase(:) = phase(:,nk1)*CONJG(phase(:,n2(nk1)))
      DO banda = 1,nbands
        DO banda1 = 1,nbands
!         rho(nk1,banda, banda1,2) = SUM((psi2(nk1,banda,:) - psi2(n2(nk1),banda1,:))**2)
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
      dphase(:) = phase(:,nk1)*CONJG(phase(:,n3(nk1)))
      DO banda = 1,nbands
        DO banda1 = 1,nbands
!         rho(nk1,banda, banda1,3) = SUM((psi2(nk1,banda,:) - psi2(n3(nk1),banda1,:))**2)
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
    DO banda = 1,nbands
      DO banda1 = 1,nbands
        DO i = 0,3
          IF (connections(nk1,banda,i) == 0 .AND. dp(nk1,banda, banda1,i) > tol1) THEN
            connections1(nk1,banda,i) = banda1
          ENDIF
        ENDDO
      ENDDO
    ENDDO
  ENDDO

  DO nk1 = 0,nks-1
    DO banda = 1,nbands
      DO banda1 = 1,nbands
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
    DO banda = 1,nbands
      WRITE(9,fmt5)nk1,banda,connections(nk1,banda,:)
    ENDDO
  ENDDO    
  CLOSE(UNIT=9)
  OPEN(UNIT=9,FILE='connections1',STATUS='UNKNOWN')
  DO nk1 = 0,nks-1
    DO banda = 1,nbands
      WRITE(9,fmt5)nk1,banda,connections1(nk1,banda,:)
    ENDDO
  ENDDO    
  CLOSE(UNIT=9)
  OPEN(UNIT=9,FILE='connections2',STATUS='UNKNOWN')
  DO nk1 = 0,nks-1
    DO banda = 1,nbands
      WRITE(9,fmt5)nk1,banda,connections2(nk1,banda,:)
    ENDDO
  ENDDO    
  CLOSE(UNIT=9)
  OPEN(UNIT=9,FILE='dp.dat',STATUS='UNKNOWN')
  DO nk1 = 0,nks-1
    DO banda = 1,nbands
      DO banda1 = 1,nbands
        WRITE(9,fmt6) nk1,banda,banda1,dp(nk1,banda, banda1,:)
      ENDDO
    ENDDO
  ENDDO
  CLOSE(UNIT=9)
  OPEN(UNIT=9,FILE='dpc.dat',STATUS='UNKNOWN')
  DO nk1 = 0,nks-1
    DO banda = 1,nbands
      DO banda1 = 1,nbands
        WRITE(9,fmt7) nk1,banda,banda1,dpc(nk1,banda, banda1,:)
      ENDDO
    ENDDO
  ENDDO
  CLOSE(UNIT=9)



! ****************************************************************************


END PROGRAM connect

