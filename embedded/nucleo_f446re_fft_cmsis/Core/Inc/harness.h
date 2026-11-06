#ifndef HARNESS_H
#define HARNESS_H

/* Called once after the HAL and USART2 are initialised. */
void harness_init(void);

/* One complete report: header, configuration, transforms, timing, memory,
 * END. Called repeatedly from the main loop. */
void harness_run(void);

#endif
