/* Route newlib's stdout to USART2, so the harness prints with the same
 * printf format strings as the host reference and the two streams diff
 * cleanly. Blocking transmit: nothing here is timed while printing. */
#include <errno.h>
#include <sys/unistd.h>
#include "main.h"

extern UART_HandleTypeDef huart2;

int _write(int fd, char *buf, int len)
{
    if (fd != STDOUT_FILENO && fd != STDERR_FILENO) {
        errno = EBADF;
        return -1;
    }
    HAL_UART_Transmit(&huart2, (uint8_t *)buf, (uint16_t)len, HAL_MAX_DELAY);
    return len;
}
