import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { MongooseModule } from '@nestjs/mongoose';
//import { DoctorsModule } from './doctors/doctors.module';
import { AuthModule } from './auth/auth.module';
import { ErrorsModule } from './errors/errors.module';
import { GlobalErrorModule } from './global-error/global-error.module';
import { UserSchema } from './models/user.schema';
import { DoctorAuthModule } from './doctor-auth/doctor-auth.module';

//testing

@Module({
  imports: [
    MongooseModule.forRoot(
      'mongodb+srv://PCOSFertilicareDB:6OjUyaiYp4X8365d@cluster0.zpgsbrp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
      { connectTimeoutMS: 300000, socketTimeoutMS: 300000 },
    ),
    MongooseModule.forFeature([{ name: 'User', schema: UserSchema }]),
    //DoctorsModule,
    AuthModule,
    ErrorsModule,
    GlobalErrorModule,
    DoctorAuthModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
