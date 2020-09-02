!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!*********************************************************************!
!*                                                                  **!
!*                         interpolation                            **!
!*                      ====================                        **!
!*                                                                  **!
!*  This program finds which wavefunctions should be interpolated   **!
!*  and which wavefunctions should be used for the interpolation.   **!
!*  Then it creates the interpolated functions using polinomial     **!
!*  interpolation
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!************************************************************************
! compiles with command
! py -m numpy.f2py -c interpolation.f90 -m interpolat
!

SUBROUTINE polinomial(npontos,x,y,x0,y0)

  IMPLICIT NONE

  INTEGER,INTENT(IN) :: npontos
  INTEGER npontos1
  INTEGER :: i, iter, alpha, m
  REAL(KIND=8),INTENT(IN) :: y(0:10), x(0:10), x0
  REAL(KIND=8),ALLOCATABLE :: p(:,:)
  REAL(KIND=8),INTENT(OUT) :: y0

  npontos1 = npontos - 1
  ALLOCATE(p(0:npontos1,0:npontos1))
  p = -1

  DO i = 0, npontos1
    p(0,i) = y(i)
  ENDDO

  DO iter = 1, npontos1
    DO alpha = 0, npontos1 - iter
      m = alpha + iter
      p(iter,alpha) = ((x0 - x(m))*p(iter-1,alpha) + (x(alpha) - x0)*p(iter-1,alpha+1))/(x(alpha) - x(m))
!      WRITE(*,*) iter, alpha, m, p(iter,alpha)
    ENDDO
  ENDDO

  y0 = p(npontos1,0)

  RETURN

END SUBROUTINE polinomial



!*********************************************************************!
SUBROUTINE interpol(nr,nk0,nb0,xx0,xx1,bx0,bx1,wfcdirectory)
!*********************************************************************!

IMPLICIT NONE

  INTEGER, INTENT(IN) :: nk0, nb0, nr
  INTEGER, DIMENSION(0:6), INTENT(IN) :: xx0, xx1, bx0, bx1

  INTEGER :: npontos, flag
  INTEGER(KIND=4) :: i, ii
  CHARACTER(LEN=15) :: str1, str2
  CHARACTER(LEN=50) :: wfcdirectory, outfile
  REAL(KIND=8),ALLOCATABLE :: psir(:,:),psii(:,:)
  REAL(KIND=8),ALLOCATABLE :: psirnewx(:,:),psiinewx(:,:)
  REAL(KIND=8),ALLOCATABLE :: psirnewy(:,:),psiinewy(:,:)
  REAL(KIND=8) :: y(0:10), x(0:10), x0
  CHARACTER(LEN=20) :: fmt1


  ALLOCATE(psir(0:6,1:nr),psii(0:6,1:nr))
  ALLOCATE(psirnewx(0:6,1:nr),psiinewx(0:6,1:nr))
  ALLOCATE(psirnewy(0:6,1:nr),psiinewy(0:6,1:nr))

  fmt1 = '(2f22.16)'
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  WRITE(*,*) ' K point, band to interpolate:', nk0, nb0
  flag = 0
  ! direction x: xx0, bx0
  npontos = 0
  x0 = 0
  DO ii = 0,6     ! Run through a line to fetch the data
    IF (xx0(ii) > -1 .AND. ii .NE. 3) THEN
      x(npontos) = ii
      WRITE(str1,*) xx0(ii)
      WRITE(str2,*) bx0(ii)
      outfile = trim(wfcdirectory)//'/k0'//trim(adjustl(str1))//'b0'//trim(adjustl(str2))//'.wfc'
      WRITE(*,*) ' Reading file: '//outfile
      OPEN(FILE=outfile,UNIT=3,STATUS='OLD')
      DO i = 1, nr
        READ(3,fmt1) psir(npontos,i),psii(npontos,i)
      ENDDO
      CLOSE(UNIT=3)
      npontos = npontos + 1
    ENDIF
  ENDDO

  IF (npontos > 1) THEN
    flag = flag + 1
    DO i = 1, nr
      y = 0.0
      DO ii = 0,npontos-1
        y(ii) = psir(ii,i)
      ENDDO
      CALL polinomial(npontos,x,y,x0,psirnewx(ii,i))
      y = 0.0
      DO ii = 0,npontos-1
        y(ii) = psii(ii,i)
      ENDDO
      CALL polinomial(npontos,x,y,x0,psiinewx(ii,i))
    ENDDO
  ENDIF

  ! direction y
  npontos = 0
  x0 = 0
  DO ii = 0,6     ! Run through a line to fetch the data
    IF (xx1(ii) > -1 .AND. ii .NE. 3) THEN
      x(npontos) = ii
      WRITE(str1,*) xx1(ii)
      WRITE(str2,*) bx1(ii)
      outfile = trim(wfcdirectory)//'/k0'//trim(adjustl(str1))//'b0'//trim(adjustl(str2))//'.wfc'
      WRITE(*,*) ' Reading file: '//outfile
      OPEN(FILE=outfile,UNIT=3,STATUS='OLD')
      DO i = 1, nr
        READ(3,fmt1) psir(npontos,i),psii(npontos,i)
      ENDDO
      CLOSE(UNIT=3)
      npontos = npontos + 1
    ENDIF
  ENDDO


  IF (npontos > 1) THEN
    flag = flag + 2
    DO i = 1, nr
      y = 0.0
      DO ii = 0,npontos-1
        y(ii) = psir(ii,i)
      ENDDO
      CALL polinomial(npontos,x,y,x0,psirnewy(ii,i))
      y = 0.0
      DO ii = 0,npontos-1
        y(ii) = psii(ii,i)
      ENDDO
      CALL polinomial(npontos,x,y,x0,psiinewy(ii,i))
    ENDDO
  ENDIF

  WRITE(str1,*) nk0
  WRITE(str2,*) nb0
  outfile = trim(wfcdirectory)//'/k0'//trim(adjustl(str1))//'b0'//trim(adjustl(str2))//'.wfc1'
  WRITE(*,*) ' Writing file: '//outfile
  OPEN(FILE=outfile,UNIT=3,STATUS='UNKNOWN')
  IF (flag == 1) THEN
    DO i = 1, nr
      WRITE(3,fmt1) psirnewx(npontos,i),psiinewx(npontos,i)
    ENDDO
  ELSEIF (flag == 2) THEN
    DO i = 1, nr
      WRITE(3,fmt1) psirnewy(npontos,i),psiinewy(npontos,i)
    ENDDO
  ELSE
    DO i = 1, nr
      WRITE(3,fmt1) (psirnewx(npontos,i)+psirnewy(npontos,i))/2.0, &
                            (psiinewx(npontos,i)+psiinewy(npontos,i))/2.0
    ENDDO
  ENDIF
  CLOSE(UNIT=3)




END SUBROUTINE interpol

