import * as mongoose from 'mongoose';
import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { HydratedDocument } from 'mongoose';

export type UserDocument = HydratedDocument<DoctorAuth>;


@Schema()
export class DoctorAuth {
  @Prop()
  registrationNumber: number;
}

export const DoctorAuthSchema = SchemaFactory.createForClass(DoctorAuth);


