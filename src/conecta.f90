!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!*********************************************************************!
!*                                                                  **!
!*                          connections                             **!
!*                      ====================                        **!
!*                                                                  **!
!*  Ricardo Mendes Ribeiro                                          **!
!*  Date: Jan, 2020                                                 **!
!*  Description: Main program that compares two wavefunctions       **!
!*             of a set and make connections                        **!
!*                                                                  **!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!************************************************************************

PROGRAM conect

IMPLICIT NONE

  INTEGER(KIND=4) :: i, j, banda, banda1
  INTEGER :: IOstatus
  INTEGER :: numero_kx, numero_ky, numero_kz
  INTEGER :: nbands, nks, nr
  INTEGER :: nkk0,nkk1
  CHARACTER(LEN=20) :: fmt1, fmt2, fmt3, fmt4, fmt5
  CHARACTER(LEN=50) :: wfcdirectory
  CHARACTER(LEN=15) :: str1, str2
  CHARACTER(LEN=50),ALLOCATABLE :: infile(:)
  REAL(KIND=8),PARAMETER :: pi2 = 6.28318530717958647688
  COMPLEX, PARAMETER :: im = (0,1)
  REAL(KIND=8),ALLOCATABLE :: psi2(:,:,:), psir(:,:,:), psii(:,:,:)
  REAL(KIND=8) :: a,b,c
  REAL(KIND=8),ALLOCATABLE :: dp(:,:,:,:), rho(:,:,:,:)
  INTEGER(KIND=4),ALLOCATABLE :: connections(:,:,:)
  REAL(KIND=8),ALLOCATABLE :: x(:), y(:), z(:)
  REAL(KIND=8),ALLOCATABLE :: kx(:), ky(:), kz(:)
  COMPLEX(KIND=8),ALLOCATABLE :: fase(:), fase0(:), fase1(:)
  COMPLEX(KIND=8) :: media

  fmt1 = '(3f14.8,3f22.16)'
  fmt2 = '(i4)'
  fmt3 = '(3i6)'
  fmt4 = '(5i6)'
  fmt5 = '(6i6)'

  wfcdirectory = 'wfc'
!  nr = 86400+1 !68544 !45504

  PRINT*, ' Enter the k-point numbers you want to compare'
  READ(*,*) nkk0, nkk1
  WRITE(*,*) ' Comparing k-points: ', nkk0, nkk1

  WRITE(*,*) ' Reading from file connections.dat'
  OPEN(UNIT=2,FILE='connections.dat',STATUS="OLD")
  READ(2,*) numero_kx, numero_ky, numero_kz
  WRITE(*,*) numero_kx, numero_ky, numero_kz
  READ(2,*) nbands
  READ(2,*) nks
  READ(2,*) nr
  nr = nr -1
  WRITE(*,*) ' Number of bands ',nbands
  WRITE(*,*) ' Number of k-points: ', nks
  WRITE(*,*) ' Size of wfc: ', nr
  WRITE(*,*)
  CLOSE(UNIT=2)

  ALLOCATE(infile(1:nbands))
  ALLOCATE(psi2(0:1,1:nbands,1:nr), psir(0:1,1:nbands,1:nr), psii(0:1,1:nbands,1:nr))
  ALLOCATE(dp(0:1,1:nbands,1:nbands,0:3), rho(0:1,1:nbands,1:nbands,0:3))
  ALLOCATE(connections(0:1,1:nbands,0:3))
  ALLOCATE(x(0:nr-1),y(0:nr-1),z(0:nr-1))
  ALLOCATE(fase(0:nr-1),fase0(0:nr-1),fase1(0:nr-1))
  ALLOCATE(kx(0:nks-1),ky(0:nks-1),kz(0:nks-1))

  OPEN(UNIT=2,FILE='wfc/rindex',STATUS='OLD')
  DO i = 0,nr-1
    READ(2,*)j,x(i),y(i),z(i)
  ENDDO
  CLOSE(UNIT=2)

  OPEN(UNIT=2,FILE='wfc/k_points',STATUS='OLD')
  DO i = 0,nks-1
    READ(2,*) kx(i), ky(i), kz(i)
  ENDDO
  CLOSE(UNIT=2)

  WRITE(*,*) kx(nkk0), ky(nkk0), kz(nkk0)
  WRITE(*,*) kx(nkk1), ky(nkk1), kz(nkk1)
  connections = 0

  DO i = 0,nr-1
    fase0(i) = EXP(im*pi2*(kx(nkk0)*x(i) + ky(nkk0)*y(i) + kz(nkk0)*z(i)))
    fase1(i) = EXP(im*pi2*(kx(nkk1)*x(i) + ky(nkk1)*y(i) + kz(nkk1)*z(i)))
    fase(i) = fase1(i)*CONJG(fase0(i))
  ENDDO
  WRITE(*,*) fase(10),fase0(10),fase1(10)

! ****************************************************************************
  WRITE(*,*)' Start reading files'
  DO banda = 1,nbands
    WRITE(str1,*) nkk0
    WRITE(str2,*) banda
    infile(banda) = trim(wfcdirectory)//'/k000'//trim(adjustl(str1))//'b000'//trim(adjustl(str2))//'.wfc'
    OPEN(FILE=infile(banda),UNIT=5,STATUS='OLD')
    i = 1
    IOstatus = 0
    DO WHILE (IOstatus == 0)
      READ(UNIT=5,FMT=fmt1,IOSTAT=IOstatus) a,b,c,psi2(0,banda,i),psir(0,banda,i),psii(0,banda,i)
      i = i + 1
    ENDDO
    CLOSE(UNIT=5)
    WRITE(str1,*) nkk1
    WRITE(str2,*) banda
    infile(banda) = trim(wfcdirectory)//'/k000'//trim(adjustl(str1))//'b000'//trim(adjustl(str2))//'.wfc'
    OPEN(FILE=infile(banda),UNIT=5,STATUS='OLD')
    i = 1
    IOstatus = 0
    DO WHILE (IOstatus == 0)
      READ(UNIT=5,FMT=fmt1,IOSTAT=IOstatus) a,b,c,psi2(1,banda,i),psir(1,banda,i),psii(1,banda,i)
      i = i + 1
    ENDDO
    CLOSE(UNIT=5)
  ENDDO
  WRITE(*,*)' Finished reading files'

! ****************************************************************************
  WRITE(*,*)' Start calculating connections'
  DO banda = 1,nbands
    DO banda1 = 1,nbands
      rho(0,banda, banda1,0) = SUM((psi2(0,banda,:) - psi2(1,banda1,:))**2)
      dp(0,banda, banda1,0) = ABS(SUM(fase(:)*CMPLX(psir(1,banda,:),psii(1,banda,:),KIND=8)*  &
                                              CMPLX(psir(0,banda1,:),-psii(0,banda1,:),KIND=8)))
      IF (dp(0,banda, banda1,0) > 0.9 .OR. (dp(0,banda, banda1,0) > 0.1 .AND.rho(0,banda, banda1,0) < 1E-5))THEN
        WRITE(*,*)banda, banda1, dp(0,banda, banda1,0),rho(0,banda, banda1,0)
      ENDIF
    ENDDO
  ENDDO

  media = (0,0)
  banda = 2
  banda1 = 2
  DO i = 0,nr-1
    IF (ABS(CMPLX(psir(0,banda1,i),psii(0,banda1,i),KIND=8)) > 1E-5) THEN
      media = media + CMPLX(psir(1,banda,i),psii(1,banda,i),KIND=8)/  &
                      CMPLX(psir(0,banda1,i),psii(0,banda1,i),KIND=8)
    ENDIF
  ENDDO
  media = media/nr
  WRITE(*,*) media

! ****************************************************************************


END PROGRAM conect

