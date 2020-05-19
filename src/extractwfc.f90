!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!*********************************************************************!
!*                                                                  **!
!*                           extractwfc                             **!
!*                      ====================                        **!
!*                                                                  **!
!*  Ricardo Mendes Ribeiro                                          **!
!*  Date: Jan, 2020                                                 **!
!*  Description: Main program that converts the wavefunction from   **!
!*             format output by QE to a specified format            **!
!*                                                                  **!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!************************************************************************

PROGRAM extractwfc

!  USE COMUM
!  USE MPI

IMPLICIT NONE

  INTEGER(KIND=4) :: i,j,k,l,ii,jj,maq,conta0,banda
  REAL(KIND=16) :: norma,deltaphase
  INTEGER :: nr1x, nr2x, nr3x, nr1, nbands
  CHARACTER(LEN=50) :: dummy, wfcdirectory, outfile
  COMPLEX*16 :: z
  REAL(KIND=16) :: tmpx, tmpy, nor
  INTEGER(KIND=4) :: nr
  INTEGER  :: limit, limite, nk1
  CHARACTER(LEN=15) :: str1, str2
  CHARACTER(LEN=20) :: fmt1
  REAL(KIND=8),ALLOCATABLE :: rx(:),ry(:),rz(:)
  REAL(KIND=16),ALLOCATABLE :: psi2(:,:),modulus(:,:),phase(:,:),psir(:,:),psii(:,:)
  LOGICAL :: flag

  NAMELIST / input / limite, nk1, wfcdirectory
! limite = 40
! wfcdirectory = 'wfc'
! nk1 = 0

  READ(*,NML=input)

  maq = 0
!  CALL INITIALIZE
  fmt1 = '(3f14.8,3f22.16)'

  IF (maq .EQ. 0) THEN
    WRITE(*,*) 'Reading from file wfck2r.mat k-point ',nk1
    OPEN(FILE='wfck2r.mat', UNIT=2,STATUS="OLD")
!   Read useless data 
    DO WHILE (dummy .ne. '# name: unkr')
      READ(2,'(A)') dummy
!      WRITE(*,*) dummy
    ENDDO
    READ(2,*) dummy
    READ(2,*) dummy
    READ(2,*) nr1x, nr2x, nr3x, nbands, j
!    WRITE(*,*) nr1x, nr2x, nr3x
  ENDIF

  norma = 1.0/SQRT(DBLE(nr1x*nr2x*nr3x))    ! Norm of wfc
  nr1 = nr1x * nr2x * nr3x        ! Total number of points in R space for wfc
  tmpx = SQRT(3.0)/(2.0*DBLE(nr1x))
  tmpy = 1.0/(2.0*DBLE(nr2x))
  limit = nr3x - limite
  WRITE(*,*) limite, limit, nr3x
!  WRITE(*,*) norma, nr1, 1/norma

  ALLOCATE(rx(1:nr1), ry(1:nr1),rz(1:nr1))
  ALLOCATE(psi2(1:nbands,1:nr1),modulus(1:nbands,1:nr1),phase(1:nbands,1:nr1))
  ALLOCATE(psir(1:nbands,1:nr1),psii(1:nbands,1:nr1))

  conta0 = 0
  DO l = 1, nr3x
    DO j = 1, nr2x
      DO i = 1, nr1x
        IF (l < limite .OR. l > limit) THEN
          conta0 = conta0 + 1
          rx(conta0) = tmpx*(i + j - 2)            ! X-coordinate
          ry(conta0) = tmpy*(i - j)                ! Y-coordinate
          rz(conta0) = DBLE(l - 1)
        ENDIF
      ENDDO
    ENDDO
  ENDDO

  flag = .TRUE.                       ! Flag to signal when modulus of wfc = 0
  DO banda = 1, nbands
    nor = 0
    conta0 = 0
    DO l = 1, nr3x
      DO j = 1, nr2x
        DO i = 1, nr1x
          READ(2,*) z
          IF (l < limite .OR. l > limit) THEN
            conta0 = conta0 + 1
            psir(banda,conta0) = REAL(z)*norma
            psii(banda,conta0) = AIMAG(z)*norma
            phase(banda,conta0) = ATAN2(psii(banda,conta0),psir(banda,conta0))
            psi2(banda,conta0) = psir(banda,conta0)**2 + psii(banda,conta0)**2
            modulus(banda,conta0) = SQRT(psi2(banda,conta0))
            nor = nor + psi2(banda,conta0)
          ENDIF
        ENDDO
      ENDDO
    ENDDO
    nr = conta0
    deltaphase = phase(banda,INT(nr1x*nr2x*1.1))
    IF (psi2(banda,INT(nr1x*nr2x*1.1)) < 1E-10) THEN    ! If modulus of wfc = 0
      flag = .FALSE.
    ENDIF
    WRITE(*,'(2i5,3f14.8,L3)') nk1,banda,SQRT(nor),psi2(banda,INT(nr1x*nr2x*1.1)),deltaphase,flag

    WRITE(str1,*) nk1
    WRITE(str2,*) banda
    outfile = trim(wfcdirectory)//'/k000'//trim(adjustl(str1))//'b000'//trim(adjustl(str2))//'.wfc'

    OPEN(FILE=outfile,UNIT=3,STATUS='UNKNOWN')

    DO i = 1, nr
      phase(banda,i) = phase(banda,i) - deltaphase
      psir(banda,i) = modulus(banda,i)*COS(phase(banda,i))
      psii(banda,i) = modulus(banda,i)*SIN(phase(banda,i))
      WRITE(3,fmt1) rx(i),ry(i),rz(i),psi2(banda,i),psir(banda,i),psii(banda,i)
    ENDDO

    CLOSE(UNIT=3)

!    write(*,*) deltaphase,banda,nr

  ENDDO
























END PROGRAM extractwfc
