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
PROGRAM interpolation
!*********************************************************************!

IMPLICIT NONE

  INTEGER(KIND=4) :: i, j, l, ii, nk, nk1, banda, banda1
  INTEGER :: ninter
  INTEGER :: flag
  INTEGER :: numero_kx, numero_ky, numero_kz
  INTEGER :: nbands, nks, nr, mbands, npontos
  INTEGER(KIND=4),ALLOCATABLE :: n0(:),n1(:),n2(:),n3(:)
  INTEGER(KIND=4),ALLOCATABLE :: planes(:,:), quality(:,:)
  INTEGER(KIND=4),ALLOCATABLE :: inter(:,:,:)    ! number, (k,band) to interpolate; 0-4 k,band around
  INTEGER(KIND=4),ALLOCATABLE :: around(:)
  INTEGER(KIND=4),ALLOCATABLE :: neig(:,:,:), neigb(:,:,:)
  CHARACTER(LEN=20),PARAMETER :: fmt1 = '(3f14.8,3f22.16)', fmt2 = '(i4)', fmt3 = '(4i6)', fmt4 = '(5i6)',fmt5 = '(i6)'
  CHARACTER(LEN=15) :: str1, str2
  CHARACTER(LEN=50) :: wfcdirectory, outfile
  REAL(KIND=8),ALLOCATABLE :: psir(:,:),psii(:,:)
  REAL(KIND=8),ALLOCATABLE :: psirnewx(:,:),psiinewx(:,:)
  REAL(KIND=8),ALLOCATABLE :: psirnewy(:,:),psiinewy(:,:)
  REAL(KIND=8) :: a,b,c,d
  REAL(KIND=8) :: y(0:10), x(0:10), x0

  mbands = 11
  wfcdirectory = 'wfc'

  WRITE(*,*) ' Reading from file connections.dat'
  OPEN(FILE='connections.dat', UNIT=2,STATUS="OLD")
  READ(2,*) numero_kx, numero_ky, numero_kz
  WRITE(*,*) numero_kx, numero_ky, numero_kz
  READ(2,*) nbands
  READ(2,*) nks
  READ(2,*) nr
  WRITE(*,*) ' Number of bands ',nbands, mbands
  WRITE(*,*) ' Number of k-points: ', nks
  WRITE(*,*) ' Size of wfc: ', nr
  WRITE(*,*)
  CLOSE(UNIT=2)
!  nr = nr - 1

  ALLOCATE(n0(0:nks),n1(0:nks),n2(0:nks),n3(0:nks))
  ALLOCATE(planes(0:nks-1,1:nbands),quality(0:nks-1,1:nbands))
  ALLOCATE(inter(1:1000,0:1,0:4))
  ALLOCATE(around(1:1000))
  ALLOCATE(neig(1:1000,-3:3,-3:3),neigb(1:1000,-3:3,-3:3))
  ALLOCATE(psir(0:6,1:nr),psii(0:6,1:nr))
  ALLOCATE(psirnewx(0:6,1:nr),psiinewx(0:6,1:nr))
  ALLOCATE(psirnewy(0:6,1:nr),psiinewy(0:6,1:nr))

! ****************************************************************************
  WRITE(*,*)' Start reading files'
  OPEN(UNIT=3,FILE='neighbors',STATUS='OLD')
  DO i = 0,nks-1
    READ(3,fmt4) nk,n0(i),n1(i),n2(i),n3(i)
  ENDDO
  CLOSE(UNIT=3)
  WRITE(*,*)' neighbors read'

  OPEN(UNIT=3,FILE='apontador',STATUS='OLD')
  DO banda = 1,nbands
    nk = -1
    DO j = 0,numero_ky-1
      DO i = 0,numero_kx-1
        nk = nk + 1
        READ(3,fmt3) nk1, planes(nk,banda), banda1, quality(nk,banda)
      ENDDO
    ENDDO
  ENDDO
  CLOSE(UNIT=3)
  WRITE(*,*)' apontador read'

! ****************************************************************************
! Classify the interpolations, based on quality array
  ninter = 0      ! Number of functions to interpolate
  inter = -1      ! k-points used for interpolation (0), and machine bands(1)
  around = 0
  WRITE(*,*)'direction 0         1         2         3         nk      classification'
  DO nk = 0,nks-1
    DO banda = 1,mbands
      IF (quality(nk,banda) == 1 .OR. quality(nk,banda) == 2) CYCLE
      ninter = ninter + 1
      inter(ninter,0,4) = nk                               ! k point to be interpolated
      inter(ninter,1,4) = planes(nk,banda)                 ! machine nr to be interpolated 
      IF (n0(nk) .GE. 0) THEN
        IF (planes(n0(nk),banda) > 0 .AND. quality(n0(nk),banda) < 5) THEN
          inter(ninter,0,0) = n0(nk)                           ! k point direction 0
          inter(ninter,1,0) = planes(n0(nk),banda)             ! machine nr direction 0
          around(ninter) = around(ninter) + 1
        ENDIF
      ENDIF
      IF (n1(nk) .GE. 0) THEN
        IF (planes(n1(nk),banda) > 0 .AND. quality(n1(nk),banda) < 5) THEN
          inter(ninter,0,1) = n1(nk)                           ! k point direction 1
          inter(ninter,1,1) = planes(n1(nk),banda)             ! machine nr direction 1
          around(ninter) = around(ninter) + 2
        ENDIF
      ENDIF
      IF (n2(nk) .GE. 0) THEN
        IF (planes(n2(nk),banda) > 0 .AND. quality(n2(nk),banda) < 5) THEN
          inter(ninter,0,2) = n2(nk)                           ! k point direction 2
          inter(ninter,1,2) = planes(n2(nk),banda)             ! machine nr direction 2
          around(ninter) = around(ninter) + 4
        ENDIF
      ENDIF
      IF (n3(nk) .GE. 0) THEN
        IF (planes(n3(nk),banda) > 0 .AND. quality(n3(nk),banda) < 5) THEN
          inter(ninter,0,3) = n3(nk)                           ! k point direction 3
          inter(ninter,1,3) = planes(n3(nk),banda)             ! machine nr direction 3
          around(ninter) = around(ninter) + 8
        ENDIF
      ENDIF
      WRITE(*,*) inter(ninter,0,1:4), around(ninter)
      WRITE(*,*) inter(ninter,1,1:4)
    ENDDO
  ENDDO
  WRITE(*,*) ' Number of functions to interpolate: ',ninter
  WRITE(*,*) ' Binary code to classify type of interpolation'
  DO i = 1,15
    WRITE(*,*)i,COUNT((around(:) == i))
  ENDDO

  neig = -1                       ! kpoint to be interpolated and surrounding k points
  neigb = -1                      ! band to be interpolated and corresponding bands around
  DO i = 1, ninter
    neig(i,0,0)   = inter(i,0,4)
    neigb(i,0,0)  = inter(i,1,4)
    neig(i,-1,0)  = inter(i,0,0)
    neigb(i,-1,0) = inter(i,1,0)
    neig(i,1,0)   = inter(i,0,2)
    neigb(i,1,0)  = inter(i,1,2)
    neig(i,0,-1)  = inter(i,0,1)
    neigb(i,0,-1) = inter(i,1,1)
    neig(i,0,1)   = inter(i,0,3)
    neigb(i,0,1)  = inter(i,1,3)
  ENDDO

  DO i = 1, ninter               ! Run through all points to be interpolated
    IF (neig(i,-1,0) > -1) THEN
      IF (n0(neig(i,-1,0)) > 0) THEN
        neig(i,-2,0) = n0(neig(i,-1,0))
        neigb(i,-2,0) = planes(n0(neig(i,-1,0)),neigb(i,-1,0))
      ENDIF
      IF (n3(neig(i,-1,0)) > 0) THEN
        neig(i,-1,1) = n3(neig(i,-1,0))
        neigb(i,-1,1) = planes(n3(neig(i,-1,0)),neigb(i,-1,0))
      ENDIF
      IF (n1(neig(i,-1,0)) > 0) THEN
        neig(i,-1,-1) = n1(neig(i,-1,0))
        neigb(i,-1,-1) = planes(n1(neig(i,-1,0)),neigb(i,-1,0))
      ENDIF
    ENDIF
    IF (neig(i,1,0) > -1) THEN
      IF (n2(neig(i,1,0)) > 0) THEN
        neig(i,2,0) = n2(neig(i,1,0))
        neigb(i,2,0) = planes(n2(neig(i,1,0)),neigb(i,1,0))
      ENDIF
      IF (n3(neig(i,1,0)) > 0) THEN
        neig(i,1,1) = n3(neig(i,1,0))
        neigb(i,1,1) = planes(n3(neig(i,1,0)),neigb(i,1,0))
      ENDIF
      IF (n1(neig(i,1,0)) > 0) THEN
        neig(i,1,-1) = n1(neig(i,1,0))
        neigb(i,1,-1) = planes(n1(neig(i,1,0)),neigb(i,1,0))
      ENDIF
    ENDIF
    IF (neig(i,0,-1) > -1) THEN
      IF (n1(neig(i,0,-1)) > 0) THEN
        neig(i,0,-2) = n1(neig(i,0,-1))
        neigb(i,0,-2) = planes(n1(neig(i,0,-1)),neigb(i,0,-1))
      ENDIF
      IF (n2(neig(i,0,-1)) > 0) THEN
        neig(i,1,-1) = n2(neig(i,0,-1))
        neigb(i,1,-1) = planes(n2(neig(i,0,-1)),neigb(i,0,-1))
      ENDIF
      IF (n0(neig(i,0,-1)) > 0) THEN
        neig(i,-1,-1) = n0(neig(i,0,-1))
        neigb(i,-1,-1) = planes(n0(neig(i,0,-1)),neigb(i,0,-1))
      ENDIF
    ENDIF
    IF (neig(i,0,1) > -1) THEN
      IF (n3(neig(i,0,1))  > 0) THEN
        neig(i,0,2) = n3(neig(i,0,1))
        neigb(i,0,2) = planes(n3(neig(i,0,1)),neigb(i,0,1))
      ENDIF
      IF (n2(neig(i,0,1))  > 0) THEN
        neig(i,1,1) = n2(neig(i,0,1))
        neigb(i,1,1) = planes(n2(neig(i,0,1)),neigb(i,0,1))
      ENDIF
      IF (n0(neig(i,0,1))  > 0) THEN
        neig(i,-1,1) = n0(neig(i,0,1))
        neigb(i,-1,1) = planes(n0(neig(i,0,1)),neigb(i,0,1))
      ENDIF
    ENDIF
  ENDDO

  DO i = 1, ninter
    IF (neig(i,-2,0) > -1) THEN
      IF (n0(neig(i,-2,0)) > 0) THEN
        neig(i,-3,0) = n0(neig(i,-2,0))
        neigb(i,-3,0) = planes(n0(neig(i,-2,0)),neigb(i,-2,0))
      ENDIF
      IF (n3(neig(i,-2,0)) > 0) THEN
        neig(i,-2,1) = n3(neig(i,-2,0))
        neigb(i,-2,1) = planes(n3(neig(i,-2,0)),neigb(i,-2,0))
      ENDIF
      IF (n1(neig(i,-2,0)) > 0) THEN
        neig(i,-2,-1) = n1(neig(i,-2,0))
        neigb(i,-2,-1) = planes(n1(neig(i,-2,0)),neigb(i,-2,0))
      ENDIF
    ENDIF
    IF (neig(i,2,0) > -1) THEN
      IF (n2(neig(i,2,0)) > 0) THEN
        neig(i,3,0) = n2(neig(i,2,0))
        neigb(i,3,0) = planes(n2(neig(i,2,0)),neigb(i,2,0))
      ENDIF
      IF (n3(neig(i,2,0)) > 0) THEN
        neig(i,2,1) = n3(neig(i,2,0))
        neigb(i,2,1) = planes(n3(neig(i,2,0)),neigb(i,2,0))
      ENDIF
      IF (n1(neig(i,2,0)) > 0) THEN
        neig(i,2,-1) = n1(neig(i,2,0))
        neigb(i,2,-1) = planes(n1(neig(i,2,0)),neigb(i,2,0))
      ENDIF
    ENDIF
    IF (neig(i,0,-2) > -1) THEN
      IF (n1(neig(i,0,-2)) > 0) THEN
        neig(i,0,-3) = n1(neig(i,0,-2))
        neigb(i,0,-3) = planes(n1(neig(i,0,-2)),neigb(i,0,-2))
      ENDIF
      IF (n2(neig(i,0,-2)) > 0) THEN
        neig(i,1,-2) = n2(neig(i,0,-2))
        neigb(i,1,-2) = planes(n2(neig(i,0,-2)),neigb(i,0,-2))
      ENDIF
      IF (n0(neig(i,0,-2)) > 0) THEN
        neig(i,-1,-2) = n0(neig(i,0,-2))
        neigb(i,-1,-2) = planes(n0(neig(i,0,-2)),neigb(i,0,-2))
      ENDIF
    ENDIF
    IF (neig(i,0,2) > -1) THEN
      IF (n3(neig(i,0,2)) > 0) THEN
        neig(i,0,3) = n3(neig(i,0,2))
        neigb(i,0,3) = planes(n3(neig(i,0,2)),neigb(i,0,2))
      ENDIF
      IF (n2(neig(i,0,2)) > 0) THEN
        neig(i,1,2) = n2(neig(i,0,2))
        neigb(i,1,2) = planes(n2(neig(i,0,2)),neigb(i,0,2))
      ENDIF
      IF (n0(neig(i,0,2)) > 0) THEN
        neig(i,-1,2) = n0(neig(i,0,2))
        neigb(i,-1,2) = planes(n0(neig(i,0,2)),neigb(i,0,2))
      ENDIF
    ENDIF
  ENDDO

  WRITE(*,*)
  DO l = 1,ninter
  WRITE(*,*)
    WRITE(*,*) ' Point nr. ',l
    DO i = -3,3
      WRITE(*,*)
      DO j = -3,3
        WRITE(*,fmt5,advance="no")neig(l,i,j)
      ENDDO
      WRITE(*,'(A10)',advance="no")'          '
      DO j = -3,3
        WRITE(*,fmt2,advance="no")neigb(l,i,j)
      ENDDO
      WRITE(*,*)
    ENDDO
  ENDDO
  WRITE(*,*)


  DO j = 1, ninter      ! Runs through all k points to be interpolated
    WRITE(*,*) ' K point, band to interpolate:',neig(j,0,0), neigb(j,0,0)
    flag = 0
    ! direction x
    IF (around(j) == 1 .OR. around(j) == 4 .OR. around(j) == 5 .OR. around(j) == 3 .OR. &
        around(j) == 6 .OR. around(j) == 7 .OR. around(j) == 9 .OR. around(j) >= 11) THEN
      npontos = 0
      x0 = 0
      DO ii = -3,3     ! Run through a line to fetch the data
        IF (neig(j,ii,0) > 0 .AND. ii .NE. 0) THEN
          x(npontos) = ii
          WRITE(str1,*) neig(j,ii,0)
          WRITE(str2,*) neigb(j,ii,0)
          outfile = trim(wfcdirectory)//'/k000'//trim(adjustl(str1))//'b000'//trim(adjustl(str2))//'.wfc'
          WRITE(*,*) ' Reading file: '//outfile
          OPEN(FILE=outfile,UNIT=3,STATUS='OLD')
          DO i = 1, nr
            READ(3,*) a,b,c,d,psir(npontos,i),psii(npontos,i)
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

    ENDIF

    ! direction y
    IF (around(j) == 2 .OR. around(j) == 8 .OR. around(j) == 3 .OR. around(j) == 6 .OR. &
        around(j) >= 9) THEN
      npontos = 0
      x0 = 0
      DO ii = -3,3     ! Run through a line to fetch the data
        IF (neig(j,0,ii) > 0 .AND. ii .NE. 0) THEN
          x(npontos) = ii
          WRITE(str1,*) neig(j,0,ii)
          WRITE(str2,*) neigb(j,0,ii)
          outfile = trim(wfcdirectory)//'/k000'//trim(adjustl(str1))//'b000'//trim(adjustl(str2))//'.wfc'
          WRITE(*,*) ' Reading file: '//outfile
          OPEN(FILE=outfile,UNIT=3,STATUS='OLD')
          DO i = 1, nr
            READ(3,*) a,b,c,d,psir(npontos,i),psii(npontos,i)
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


    ENDIF

    WRITE(str1,*) neig(j,0,0)
    WRITE(str2,*) neigb(j,0,0)
    outfile = trim(wfcdirectory)//'/k000'//trim(adjustl(str1))//'b000'//trim(adjustl(str2))//'.wfc1'
    WRITE(*,*) ' Writing file: '//outfile
    OPEN(FILE=outfile,UNIT=3,STATUS='UNKNOWN')
    IF (flag == 1) THEN
      DO i = 1, nr
        WRITE(3,fmt1) a,b,c,d,psirnewx(npontos,i),psiinewx(npontos,i)
      ENDDO
    ELSEIF (flag == 2) THEN
      DO i = 1, nr
        WRITE(3,fmt1) a,b,c,d,psirnewy(npontos,i),psiinewy(npontos,i)
      ENDDO
    ELSE
      DO i = 1, nr
        WRITE(3,fmt1) a,b,c,d,(psirnewx(npontos,i)+psirnewy(npontos,i))/2.0, &
                              (psiinewx(npontos,i)+psiinewy(npontos,i))/2.0
      ENDDO
    ENDIF
    CLOSE(UNIT=3)



  ENDDO

END PROGRAM interpolation

