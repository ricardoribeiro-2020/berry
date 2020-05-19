!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!*********************************************************************!
!*                                                                  **!
!*                          find_sym                                **!
!*                      ====================                        **!


PROGRAM find_sym

IMPLICIT NONE
  INTEGER(KIND=4) :: i, j, banda, banda1
  INTEGER :: numero_kx, numero_ky, numero_kz
  INTEGER :: nbands, nks, nr, nkk0
  CHARACTER(LEN=50) :: wfcdirectory
  REAL(KIND=8),ALLOCATABLE :: kx(:), ky(:), kz(:)
  REAL(KIND=8),ALLOCATABLE :: eigenvalues(:,:)
  REAL(KIND=8),ALLOCATABLE :: sym(:,:)
  INTEGER :: a
  INTEGER,ALLOCATABLE :: contagem(:)

  wfcdirectory = 'wfc'
  PRINT*, ' Enter the k-point number you want to compare'
  READ(*,*) nkk0
  WRITE(*,*) ' Comparing with k-point: ', nkk0

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

  ALLOCATE(kx(0:nks-1),ky(0:nks-1),kz(0:nks-1))
  ALLOCATE(eigenvalues(0:nks-1,1:nbands),sym(0:nks-1,1:nbands))
  ALLOCATE(contagem(0:nks-1))

  OPEN(UNIT=2,FILE='wfc/k_points',STATUS='OLD')
  DO i = 0,nks-1
    READ(2,*) kx(i), ky(i), kz(i)
  ENDDO
  CLOSE(UNIT=2)

  OPEN(UNIT=2,FILE='wfc/eigenvalues',STATUS='OLD')
  DO i = 0,nks-1
    READ(2,*) a,eigenvalues(i,:) 
!   WRITE(*,*) a
!   WRITE(*,*) eigenvalues(i,:)
  ENDDO

  contagem = 0
  DO i = 0,nks-1
    DO j = 1,nbands
      sym(i,j) = eigenvalues(i,j)/eigenvalues(nkk0,j)
      IF (sym(i,j) > 0.99 .AND. sym(i,j) < 1.01) THEN
        contagem(i) = contagem(i) + 1
!        WRITE(*,*)i,j,sym(i,j)
      ENDIF
    ENDDO
  ENDDO

  DO i = 0,nks-1
    IF (contagem(i) > nbands-5) THEN
      WRITE(*,*)i,contagem(i)
      DO j = 1,nbands
        IF (sym(i,j) < 0.99 .OR. sym(i,j) > 1.01) THEN
          WRITE(*,*) j,sym(i,j)
        ENDIF
      ENDDO
    ENDIF
  ENDDO


END PROGRAM find_sym

