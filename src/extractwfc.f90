!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!*********************************************************************!
!*                                                                  **!
!*                           extractwfc                             **!
!*                      ====================                        **!
!*                                                                  **!
!*  Ricardo Mendes Ribeiro                                          **!
!*  Date: Jun, 2020                                                 **!
!*  Description: Main program that converts the wavefunction from   **!
!*             format output by QE to a specified format            **!
!*                                                                  **!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!************************************************************************

PROGRAM extractwfc

!  USE MPI

IMPLICIT NONE

  INTEGER(KIND=4) :: i,j,k,l,ii,jj,maq,conta0,banda
  REAL(KIND=8) :: deltaphase
  INTEGER :: nr1x, nr2x, nr3x, nr1, nbands
  CHARACTER(LEN=50) :: dummy, wfcdirectory, outfile
  COMPLEX*8 :: z
  INTEGER(KIND=4) :: nr, rpoint
  INTEGER  :: nk,nb,npr
  CHARACTER(LEN=15) :: str1, str2
  CHARACTER(LEN=20) :: fmt1
  REAL(KIND=8),ALLOCATABLE :: modulus(:),phase(:),psir(:),psii(:)
  LOGICAL :: flag

  NAMELIST / input / nk, nb, npr, wfcdirectory
  READ(*,NML=input)

  maq = 0
!  CALL INITIALIZE
  fmt1 = '(2f22.16)'

  IF (maq .EQ. 0) THEN
!    WRITE(*,*) 'Reading from file wfck2r.mat k-point ',nk,' band ',nb
    OPEN(FILE='wfck2r.mat', UNIT=2,STATUS="OLD")
!   Read useless data 
    DO WHILE (dummy .ne. '# name: unkr')
      READ(2,'(A)') dummy
    ENDDO
    READ(2,*) dummy
    READ(2,*) dummy
    READ(2,*) nr1x, nr2x, nr3x, nbands, j
!    WRITE(*,*) 'Size of wavefunction in real space',nr1x, nr2x, nr3x
  ENDIF

  nr1 = nr1x * nr2x * nr3x        ! Total number of points in R space for wfc

  ALLOCATE(modulus(1:nr1),phase(1:nr1))
  ALLOCATE(psir(1:nr1),psii(1:nr1))

  flag = .TRUE.                       ! Flag to signal when modulus of wfc = 0
  conta0 = 0
  DO l = 1, nr3x
    DO j = 1, nr2x
      DO i = 1, nr1x
        READ(2,*) z                   ! Reads data from file: cannot be parallelized
        conta0 = conta0 + 1
        psir(conta0) = REAL(z)
        psii(conta0) = AIMAG(z)
        phase(conta0) = ATAN2(psii(conta0),psir(conta0))
        modulus(conta0) = ABS(z)
      ENDDO
    ENDDO
  ENDDO
  nr = conta0

! Finished reading file

  IF (maq .EQ. 0) THEN
    rpoint = INT(nr1x*nr2x*1.1)
    deltaphase = phase(rpoint)
    IF (modulus(rpoint) < 1E-5) THEN    ! If modulus of wfc = 0
      flag = .FALSE.
    ENDIF
    WRITE(*,'(2i5,2f14.8,L3)') nk,nb,modulus(rpoint),deltaphase,flag
 
    WRITE(str1,*) nk
    WRITE(str2,*) nb
    outfile = trim(wfcdirectory)//'/k0'//trim(adjustl(str1))//'b0'//trim(adjustl(str2))//'.wfc'

    OPEN(FILE=outfile,UNIT=3,STATUS='UNKNOWN')
 
    DO i = 1, nr
      phase(i) = phase(i) - deltaphase
      psir(i) = modulus(i)*COS(phase(i))
      psii(i) = modulus(i)*SIN(phase(i))
      WRITE(3,fmt1) psir(i),psii(i)    ! Writes data to file: cannot be parallelized
    ENDDO
 
    CLOSE(UNIT=3)
  ENDIF

!  write(*,*) deltaphase,banda,nr




END PROGRAM extractwfc
