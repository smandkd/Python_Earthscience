PROGRAM circle
        IMPLICIT NONE

        INTERFACE
                FUNCTION Area_Circle(r)
                        REAL :: area_circle
                        REAL, INTENT(IN) :: r
                END FUNCTION Area_Circle
        END INTERFACE

        REAL :: radius

        write(*, '(A)', ADVANCE= "NO") &
                "Enter the radius of the circle: "
        read(*, *) radius

        write(*, 100) "Area of circle with radius", radius, &
        " is", Area_Circle(radius)
        100 format (A, 2x, F6.2, A, 2x, F11.2)

END PROGRAM circle

SUBROUTINE Compute_Area(r, Area)
        IMPLICIT NONE
        REAL, INTENT(IN) :: r
        REAL, INTENT(OUT) :: Area

        REAL, PARAMETER :: Pi = 3.1415927
        Area = Pi * r * r
END SUBROUTINE Compute_Area

FUNCTION Area_Circle(r)
        IMPLICIT NONE
        REAL :: area_circle
        REAL, INTENT(IN) :: r
        REAL, PARAMETER :: Pi = 3.1415927

        area_circle = Pi * r * r
END FUNCTION Area_Circle